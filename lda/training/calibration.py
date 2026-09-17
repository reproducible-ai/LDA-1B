"""Optional synchronized timing for independent, early-stopped training points.

The ordinary trainer supplies synchronization and checkpoint/evaluation callbacks.
The full scheduler configuration is never changed by this module.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path


class CalibrationTiming:
    def __init__(
        self,
        *,
        stop_steps,
        full_steps,
        warmup_steps,
        path,
        synchronize,
        is_main,
        completed_steps=0,
        clock=time.monotonic,
    ):
        if not 0 <= warmup_steps < stop_steps < full_steps or completed_steps != 0:
            raise ValueError("calibration requires a fresh point shorter than the full schedule")
        self.stop_steps, self.full_steps = stop_steps, full_steps
        self.warmup_steps, self.path = warmup_steps, Path(path)
        self.synchronize, self.is_main, self.clock = synchronize, is_main, clock
        self.completed_steps = 0
        self.boundaries = []
        self.losses = []
        self.checkpoint_seconds = None
        self.evaluation_seconds = None
        self.finished = None
        if is_main:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.journal = self.path.with_suffix(self.path.suffix + ".jsonl").open("x")
        else:
            self.journal = None
        self.event("start", fullSteps=full_steps, requestedSteps=stop_steps)

    def event(self, name, **data):
        if self.journal:
            line = json.dumps({"event": name, **data}, allow_nan=False)
            self.journal.write(line + "\n")
            self.journal.flush()
            os.fsync(self.journal.fileno())
            print("CALIBRATION_TRAINER_EVENT=" + line, flush=True)

    def begin(self):
        self.synchronize()
        self.boundaries.append(self.clock())

    def update(self, step, metrics):
        if step != self.completed_steps + 1 or step > self.stop_steps:
            raise ValueError("updates must be contiguous, with no resume or excess steps")
        values = [float(value) for key, value in metrics.items() if "loss" in key]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("every optimizer update must have finite training losses")
        self.synchronize()
        self.completed_steps = step
        now = self.clock()
        self.boundaries.append(now)
        self.losses.append(values)
        self.event("update", step=step, monotonicSeconds=now, losses=values)
        if step == self.stop_steps:
            self.finished = now
            return True
        return False

    def measure(self, name, action):
        if self.finished is None or name not in {"checkpoint", "evaluation"}:
            raise ValueError("overhead may only be measured after the training interval")
        self.synchronize()
        start = self.clock()
        action()
        self.synchronize()
        duration = self.clock() - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid overhead duration")
        setattr(self, name + "_seconds", duration)
        self.event(name, seconds=duration)

    def finish(self):
        if self.finished is None or self.checkpoint_seconds is None:
            raise ValueError("incomplete training or checkpoint")
        if len(self.boundaries) != self.stop_steps + 1:
            raise ValueError("incomplete update timing")
        result = {
            "fullSteps": self.full_steps,
            "completedSteps": self.completed_steps,
            "trainingFinishedMonotonicSeconds": self.finished,
            "steadyUpdates": self.stop_steps - self.warmup_steps,
            "steadySeconds": self.finished - self.boundaries[self.warmup_steps],
            "checkpointSeconds": self.checkpoint_seconds,
            "evaluationSeconds": self.evaluation_seconds,
            "finalLoss": self.losses[-1][0],
        }
        if result["steadySeconds"] <= 0:
            raise ValueError("invalid steady-state duration")
        if self.is_main:
            with self.path.open("x") as stream:
                json.dump(result, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            self.event("complete", **result)
            self.journal.close()
        return result


def timing_for_trainer(trainer):
    settings = trainer.config.get("calibration")
    if not settings:
        return None
    cfg = trainer.config
    stop = int(settings.stop_steps)
    if (
        cfg.trainer.is_resume
        or stop >= cfg.trainer.eval_interval
        or stop >= cfg.trainer.save_interval
        or trainer.accelerator.gradient_accumulation_steps != 1
        or trainer.accelerator.num_processes != int(settings.world_size)
    ):
        raise ValueError("unsupported calibration resume, cadence or distributed configuration")

    def synchronize():
        import torch

        torch.cuda.synchronize()
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()

    return CalibrationTiming(
        stop_steps=stop,
        full_steps=cfg.trainer.max_train_steps,
        warmup_steps=int(settings.exclude_first_updates),
        path=settings.timing_path,
        synchronize=synchronize,
        is_main=trainer.accelerator.is_main_process,
        completed_steps=trainer.completed_steps,
    )
