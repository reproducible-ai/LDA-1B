"""Explicit shape-only adapter for the private sim_pick_place training canary."""


def adapt_demo_canary_examples(examples, *, state_dim, action_dim, num_embodiments):
    # The committed demo identifies franka_robotiq; the upstream Franka slot is 4.
    # This reuses a checkpoint slot and does not add or resize model parameters.
    if state_dim != 58 or action_dim != 29 or num_embodiments <= 4:
        raise ValueError("Demo canary requires state/action widths 58/29 and Franka slot 4")
    adapted = []
    for example in examples:
        if example["embodiment_id"] != 32 or example["assigned_task"] != "policy":
            raise ValueError("Demo canary expects NEW_EMBODIMENT policy samples")
        state = example["state"]
        if hasattr(state, "tolist"):
            state = state.tolist()
        if not state or any(len(row) != 12 for row in state):
            raise ValueError("Demo canary expects 12-wide source state")
        if any(len(row) != action_dim for row in example["action"]):
            raise ValueError("Demo canary expects loader-padded 29-wide actions")
        adapted.append(dict(example, state=[list(row) + [0.0] * 46 for row in state], embodiment_id=4))
    return adapted
