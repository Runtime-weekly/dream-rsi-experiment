import json
import os
from pathlib import Path
import select
import selectors
import subprocess
import sys
import time


class Worker:
    def __init__(self, source, kind, timeout=20):
        self.timeout = timeout
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        self.proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).with_name("worker.py")),
                                      str(Path(source).resolve()), kind],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        self.selector.register(self.proc.stdout, selectors.EVENT_READ)
        os.set_blocking(self.proc.stdin.fileno(), False)
        try:
            if not self.read().get("ready"):
                raise RuntimeError("Sandbox worker did not become ready")
        except BaseException:
            self.close()
            raise

    def read(self):
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("Sandbox worker timed out")
            chunk = os.read(self.proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Sandbox worker exited without a result")
            self.buffer += chunk
            if len(self.buffer) > 4 * 1024**2:
                raise ValueError("Worker output exceeds limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def ask(self, observation):
        payload = (json.dumps(observation, allow_nan=False) + "\n").encode()
        deadline = time.monotonic() + self.timeout
        offset = 0
        while offset < len(payload):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [self.proc.stdin], [], remaining)[1]:
                raise TimeoutError("Sandbox worker stopped reading input")
            try:
                offset += os.write(self.proc.stdin.fileno(), payload[offset:])
            except BlockingIOError:
                continue
        response = self.read()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response["result"]

    def choose(self, observation):
        return self.ask(observation)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait(timeout=5)
        self.selector.close()
        self.proc.stdin.close()
        self.proc.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
