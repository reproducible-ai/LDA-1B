"""LDA adapter for the pinned, restart-safe public-canary supervisor."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SUPERVISOR_COMMIT = "c084d6eab090a2b5822db474d790a00dfe7fae5d"


def load_runtime(plan):
    source = Path(plan["supervisorWorkspace"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if head != SUPERVISOR_COMMIT or plan["supervisorCommit"] != head:
        raise RuntimeError("Public supervisor source must match its tested immutable revision")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True).strip():
        raise RuntimeError("Public supervisor has uncommitted changes")
    sys.path.append(str(source / ".treqs/scripts"))
    import public_canary_supervisor as runtime
    return runtime


def build_actions(runtime, plan, root):
    from lda_public_verification import prepare_repository, verify_release, publish_notes

    class LdaActions(runtime.Actions):
        def prepare(self):
            target = next(x for x in self.cli("compute", "targets", "list", "--owner", "reproducible-ai")
                          if x["id"] == plan["targetId"])
            if (not target.get("autoShutdownEnabled") or target.get("idleTimeoutMinutes", 999) > 15
                    or target["resources"]["instanceType"] != "g7e.2xlarge"
                    or target["resources"]["amiId"] != plan["amiId"]):
                raise RuntimeError("Target does not match the authorized recipe and shutdown policy")
            for directory, expected in ((plan["sourceWorkspace"], plan["sourceCommit"]),
                                        (plan["controlWorkspace"], plan["sourceCommit"]),
                                        (plan["harnessWorkspace"], plan["harnessCommit"])):
                if self.command(["git", "rev-parse", "HEAD"], cwd=directory).strip() != expected:
                    raise RuntimeError("Pinned source revision changed")
                if self.command(["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory).strip():
                    raise RuntimeError("Pinned source has uncommitted changes")
            remote = self.command(["git", "ls-remote", "origin", f"refs/heads/{plan['sourceBranch']}"],
                                  cwd=plan["sourceWorkspace"])
            if remote.split()[0] != plan["sourceCommit"]:
                raise RuntimeError("Remote source branch moved")
            prepare_repository(plan, root)

        def create(self):
            return self.cli("tr", "create", "--title", plan["title"], "--status", "open",
                            "--workflow-path", ".treqs/workflows/robocasa-demo-canary.yaml",
                            "--compute-target", plan["targetId"], "--source-branch", plan["sourceBranch"],
                            "--source-commit", plan["sourceCommit"], "--lineage-mode", "public", "--yes")

        def verify(self, state):
            job = self.job(state["jobId"])
            tasks = self.cli("jobs", "tasks", state["jobId"])
            if isinstance(tasks, dict):
                tasks = tasks["tasks"]
            if len(tasks) != 7 or any(t["status"].upper() != "COMPLETED" for t in tasks):
                raise RuntimeError("All seven tasks must be completed")
            if job.get("lineagePublicationMode") != "public":
                raise RuntimeError("Job lineage is not public")
            if job.get("lineagePublicationStatus") != "published":
                if job.get("lineagePublicationStatus") == "failed" and state.get("republishAttempts", 0) < 3:
                    state["republishAttempts"] = state.get("republishAttempts", 0) + 1
                    runtime.atomic_json(root / "state.json", state)
                    self.cli("jobs", "republish-lineage", state["jobId"])
                raise RuntimeError("Waiting for public lineage publication")
            if self.logs(state["jobId"])["hasMore"]:
                raise RuntimeError("Draining complete workload logs before verification")
            return verify_release(plan, root, job)

        def publish(self, state):
            return publish_notes(plan, root, state, self)

    return LdaActions(plan, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    plan = json.loads(args.plan.read_text())
    root = args.plan.resolve().parent
    runtime = load_runtime(plan)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervisor = runtime.Supervisor(plan, root, build_actions(runtime, plan, root))
        while True:
            try:
                if not supervisor.tick():
                    break
            except Exception as exc:
                supervisor.error(exc)
            if args.once:
                break
            time.sleep(20)


if __name__ == "__main__":
    main()
