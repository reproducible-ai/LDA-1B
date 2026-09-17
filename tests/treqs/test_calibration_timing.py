from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from lda.training.calibration import CalibrationTiming

ROOT = Path(__file__).resolve().parents[2]


def timing(tmp_path, **kwargs):
    return CalibrationTiming(
        stop_steps=3,
        full_steps=300000,
        warmup_steps=1,
        path=tmp_path / "timing.json",
        synchronize=lambda: None,
        is_main=True,
        **kwargs,
    )


def test_rejects_resume_nonfinite_and_missing_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="fresh"):
        timing(tmp_path, completed_steps=1)
    t = timing(tmp_path)
    t.begin()
    with pytest.raises(ValueError, match="contiguous"):
        t.update(2, {"loss": 1})
    with pytest.raises(ValueError, match="finite"):
        t.update(1, {"loss": float("nan")})
    with pytest.raises(ValueError, match="incomplete"):
        t.finish()


def test_synchronization_excludes_checkpoint_and_eval(tmp_path):
    ticks = iter([0, 1, 3, 6, 10, 14, 20, 27])
    t = timing(tmp_path, clock=lambda: next(ticks))
    t.begin()
    for step in range(1, 4):
        assert t.update(step, {"loss": 2.0}) == (step == 3)
    t.measure("checkpoint", lambda: None)
    t.measure("evaluation", lambda: None)
    result = t.finish()
    assert result["steadyUpdates"] == 2
    assert result["steadySeconds"] == 5
    assert result["trainingFinishedMonotonicSeconds"] == 6
    assert result["checkpointSeconds"] == 4
    assert result["evaluationSeconds"] == 7


def test_production_loop_stops_real_optimizer_without_shortening_schedule(tmp_path):
    # Importing the CUDA-only entrypoint starts DeepSpeed. Compile its actual loop
    # unchanged, and supply a tiny CPU optimizer to exercise control flow locally.
    tree = ast.parse((ROOT / "lda/training/train_LDA.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLATrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")

    class Progress:
        def update(self, n):
            pass

        def set_postfix(self, values):
            pass

    t = timing(tmp_path)
    namespace = {"time": time, "tqdm": lambda *a, **k: Progress(), "timing_for_trainer": lambda _: t}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<production-train-loop>", "exec"), namespace)
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300000)
    calls = []

    def update(batch):
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return {"loss": loss.item()}

    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"max_train_steps": 300000, "eval_interval": 1000, "save_interval": 10000},
                "calibration": {"measure_evaluation": True},
            }
        ),
        completed_steps=0,
        accelerator=SimpleNamespace(is_local_main_process=True, sync_gradients=True),
        _log_training_config=lambda: None,
        _create_data_iterators=lambda: None,
        _get_next_batch=lambda: None,
        _train_step=update,
        _log_metrics=lambda m: None,
        _save_checkpoint=lambda: calls.append("checkpoint"),
        eval_action_model=lambda m: calls.append("evaluation"),
        _finalize_training=lambda: calls.append("finalize"),
    )
    namespace["train"](trainer)
    assert trainer.completed_steps == scheduler.last_epoch == 3
    assert scheduler.T_max == 300000
    assert parameter.item() < 2
    assert calls == ["checkpoint", "evaluation", "finalize"]
    assert json.loads((tmp_path / "timing.json").read_text())["completedSteps"] == 3


def test_factory_accepts_wrapped_config_and_rejects_wrong_world_size(tmp_path, monkeypatch):
    from lda.training.calibration import timing_for_trainer
    from lda.training.trainer_utils.config_tracker import wrap_config

    cfg = wrap_config(
        OmegaConf.create(
            {
                "trainer": {
                    "is_resume": False,
                    "max_train_steps": 300000,
                    "eval_interval": 1000,
                    "save_interval": 10000,
                },
                "calibration": {
                    "stop_steps": 100,
                    "exclude_first_updates": 20,
                    "world_size": 8,
                    "timing_path": str(tmp_path / "timing.json"),
                },
            }
        )
    )
    accelerator = SimpleNamespace(
        gradient_accumulation_steps=1, num_processes=1, is_main_process=True, wait_for_everyone=lambda: None
    )
    trainer = SimpleNamespace(config=cfg, accelerator=accelerator, completed_steps=0)
    with pytest.raises(ValueError, match="distributed"):
        timing_for_trainer(trainer)
    accelerator.num_processes = 8
    result = timing_for_trainer(trainer)
    assert result.full_steps == 300000
    result.journal.close()


def test_two_real_ranks_wait_for_the_slowest_rank(tmp_path):
    import os
    import subprocess
    import sys

    worker = tmp_path / "worker.py"
    worker.write_text("""import json,sys,time
from pathlib import Path
import torch.distributed as dist
import torch.multiprocessing as mp
from lda.training.calibration import CalibrationTiming

def rank_main(rank,root):
    root=Path(root)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method='file://'+str(root/'rendezvous'))
    timing=CalibrationTiming(stop_steps=3,full_steps=300000,warmup_steps=1,path=root/'timing.json',synchronize=dist.barrier,is_main=rank==0)
    timing.begin()
    for step in range(1,4):
        if rank==1:time.sleep(.08)
        timing.update(step,{'loss':1.0})
    timing.measure('checkpoint',lambda:time.sleep(.02))
    result=timing.finish()
    assert result['steadySeconds']>=.14
    assert result['completedSteps']==3
    dist.destroy_process_group()

if __name__=='__main__':mp.spawn(rank_main,args=(sys.argv[1],),nprocs=2,join=True)
""")
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run(
        [sys.executable, str(worker), str(tmp_path)], env=env, capture_output=True, text=True, timeout=40
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "timing.json").read_text())["completedSteps"] == 3


def test_actual_launch_arguments_reach_the_trainer_parser(tmp_path, monkeypatch):
    from scripts.calibration_robocasa import run_point

    tree = ast.parse((ROOT / "lda/training/trainer_utils/trainer_tools.py").read_text())
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalize_dotlist_args")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<production-cli-parser>", "exec"), namespace)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA A100-SXM4-40GB")
    captured = []
    monkeypatch.setattr("scripts.calibration_robocasa.os.execv", lambda exe, args: captured.append(args))
    for point, steps in [("p1", 100), ("p2", 200), ("p3", 400)]:
        run_point(
            SimpleNamespace(
                plan=ROOT / ".treqs/calibration/plan.json",
                config=ROOT / ".treqs/calibration/resolved-config.json",
                output=tmp_path,
                point=point,
            )
        )
        command = captured[-1]
        overrides = OmegaConf.from_dotlist(
            namespace["normalize_dotlist_args"](command[command.index("--config_yaml") + 2 :])
        )
        assert overrides.calibration.stop_steps == steps
        assert overrides.calibration.world_size == 8
        assert overrides.calibration.measure_evaluation == (point == "p1")
        assert overrides.run_id == point
