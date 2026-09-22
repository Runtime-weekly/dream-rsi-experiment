"""Small line protocol. Generated code never runs in the scoring process."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

# Loaded before confinement; worker is launched by file path, not cwd imports.
from sandbox import confine


def main():
    source, kind = sys.argv[1:3]
    confine(source)
    spec = importlib.util.spec_from_file_location("candidate", source)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    print(json.dumps({"ready": True}), flush=True)
    loaded = None
    for line in sys.stdin:
        request = json.loads(line)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                if kind == "policy":
                    result = module.choose(request)
                elif "load" in request:
                    loaded = tuple(np.asarray(a, dtype=float) for a in request["load"])
                    result = "loaded"
                else:
                    # Copies prevent one repetition from modifying another's inputs.
                    result = np.asarray(module.solve_path(*(a.copy() for a in loaded)), dtype=float).tolist()
            print(json.dumps({"result": result}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)[:1000]}), flush=True)


if __name__ == "__main__":
    main()

