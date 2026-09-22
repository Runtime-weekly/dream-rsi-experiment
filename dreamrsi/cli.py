import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import signal
import sys
from .core import save_json
from .engine import Experiment, ROOT, source_hashes
from .model import Ollama
from .upstream import check


def validate_config(config):
    for key in ("rounds", "online_rounds", "replay_rounds", "workers", "policy_versions",
                "max_output_tokens", "model_timeout_seconds"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("cost_weight", "parallel_weight"):
        if not isinstance(config.get(key), (int, float)) or not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(key + " must be finite and nonnegative")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")


def main():
    parser = argparse.ArgumentParser(description="Independent reconstruction of Dream-RSI section 3")
    sub = parser.add_subparsers(dest="command", required=True)
    upstream = sub.add_parser("check-upstream", help="Read official release status; never install code")
    upstream.add_argument("--out", default=str(ROOT / "state/upstream.json"))
    status = sub.add_parser("status")
    status.add_argument("directory")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "check-upstream":
        print(json.dumps(check(args.out), indent=2))
        return
    if args.command == "status":
        directory = Path(args.directory)
        name = "report.json" if (directory / "report.json").exists() else "status.json"
        print((directory / name).read_text())
        return
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another process owns this experiment")
        manifest_path = directory / "manifest.json"
        if manifest_path.exists() and not args.resume:
            raise SystemExit("Existing experiment: use --resume or a new output directory")
        if args.resume and not manifest_path.exists():
            raise SystemExit("No checkpoint to resume")
        model = Ollama(config["endpoint"], config["model"], config["max_output_tokens"],
                       config["model_timeout_seconds"])
        identity = {"config": config, "model": model.identity, "source_hashes": source_hashes(),
                    "python": sys.version, "dependencies": {name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "scikit-learn")}}
        if args.resume:
            saved = json.loads(manifest_path.read_text())
            if saved["identity"] != identity:
                raise SystemExit("Model, source, evaluator, dependencies or configuration changed; start a new experiment")
        else:
            upstream = check(ROOT / "state/upstream.json")
            save_json(directory / "upstream.json", upstream)
            print("Official release: " + upstream["status"], flush=True)
            save_json(manifest_path, {"identity": identity, "started_at": datetime.now(timezone.utc).isoformat()})
        if (directory / "report.json").exists():
            print("Already complete: " + str(directory / "report.json"))
            return
        stopping = {"value": False}
        def pause(signum, frame):
            stopping["value"] = True
            print("Pause requested; finishing and saving the current operation.", flush=True)
        signal.signal(signal.SIGINT, pause)
        signal.signal(signal.SIGTERM, pause)
        save_json(directory / "status.json", {"status": "running", "model": model.identity})
        try:
            result = Experiment(directory, config, model, lambda: stopping["value"]).run()
            save_json(directory / "status.json", {"status": "completed", "report": str(directory / "report.json")})
            print(json.dumps(result, indent=2))
        except InterruptedError as exc:
            save_json(directory / "status.json", {"status": "paused", "reason": str(exc)})
        except BaseException as exc:
            save_json(directory / "status.json", {"status": "error", "reason": str(exc)})
            raise


if __name__ == "__main__":
    main()

