import ast
import json
from pathlib import Path
import re
import time
import urllib.request
from .core import digest, save_json


def request_json(url, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def extract_source(text, function):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    options = blocks + [text]
    for source in options:
        try:
            tree = ast.parse(source.strip())
            if any(isinstance(n, ast.FunctionDef) and n.name == function for n in tree.body):
                if function == "solve_path":
                    for n in ast.walk(tree):
                        if isinstance(n, ast.Import) and any(a.name.startswith("sklearn") for a in n.names):
                            raise ValueError("The candidate cannot import the reference solver")
                        if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("sklearn"):
                            raise ValueError("The candidate cannot import the reference solver")
                return source.strip() + "\n", ast.get_docstring(tree) or "No proposal docstring supplied"
        except SyntaxError:
            continue
    raise ValueError("No complete Python source defining " + function)


class Ollama:
    def __init__(self, url, name, tokens=4096, timeout=240):
        if not url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("This local reconstruction accepts only a local Ollama endpoint")
        self.url, self.name, self.tokens, self.timeout = url.rstrip("/"), name, tokens, timeout
        self.identity = self.inspect()

    def inspect(self):
        models = request_json(self.url + "/api/tags")["models"]
        model = next((m for m in models if m["name"] == self.name), None)
        if model is None:
            raise ValueError("Model must already be installed: " + self.name)
        return {"name": self.name, "digest": model["digest"], "details": model["details"]}

    def generate(self, prompt, directory, function, seed):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        request_path, response_path = directory / "request.json", directory / "response.json"
        if len(prompt) > 180000:
            raise ValueError("Full history exceeds local prompt limit; stopping without truncation")
        if response_path.exists():
            saved_request = json.loads(request_path.read_text())
            if saved_request["prompt_sha256"] != digest(prompt):
                raise RuntimeError("Saved request differs; refuse to reuse another response")
            response = json.loads(response_path.read_text())
        else:
            if request_path.exists():
                raise RuntimeError("Unconfirmed model request exists; do not automatically repeat it: " + str(directory))
            if self.inspect() != self.identity:
                raise RuntimeError("Model identity changed during experiment")
            save_json(request_path, {"prompt": prompt, "prompt_sha256": digest(prompt),
                                    "model": self.identity, "seed": seed})
            start = time.monotonic()
            try:
                answer = request_json(self.url + "/api/chat", {
                    "model": self.name, "stream": False, "think": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.4, "num_predict": self.tokens, "seed": seed},
                }, timeout=self.timeout)
                response = {"answer": answer, "seconds": time.monotonic() - start}
            except Exception as exc:
                response = {"error": type(exc).__name__ + ": " + str(exc), "seconds": time.monotonic() - start}
            save_json(response_path, response)
        if "error" in response:
            raise RuntimeError(response["error"])
        answer = response["answer"]
        source, proposal = extract_source(answer.get("message", {}).get("content", ""), function)
        if answer.get("done_reason") == "length":
            raise ValueError("Model output was truncated; not accepting incomplete source")
        return source, proposal, {"model_seconds": response["seconds"], "model_calls": 1,
                                  "input_tokens": answer.get("prompt_eval_count", 0),
                                  "output_tokens": answer.get("eval_count", 0)}

