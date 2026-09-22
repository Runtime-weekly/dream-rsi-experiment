# Dream-RSI: the RUNTIME experiment

**An independent, small reconstruction of the search-and-replay idea shown in our video.**
This is RUNTIME's implementation, not the paper authors' code and not a replication
of their published performance.

[Watch the video](https://youtu.be/opBzkTGTx8E) · [Read the results](RESULTS.md) ·
[Method and differences](research/METHOD.md) · [RUNTIME tutorials](https://github.com/Runtime-weekly/runtime-tutorials)

## What this does

A fixed coding model proposes numerical solver code. A separate search policy
chooses which attempts to continue. The program records a tree of attempts,
replays alternative policies against the recorded outcomes, and retains the best
policy—including the original policy when results tie.

**In our recorded live run, both proposed policies tied the original. Neither was
adopted.** The loop worked; the run did not demonstrate better exploration.

## Choose your path

| Your goal | Start here |
| --- | --- |
| Understand the episode without installing software | [Results and plain-language explanation](RESULTS.md) |
| Check the code locally without a model | Run the tests below; they use synthetic inputs and model fixtures |
| Run a new model-backed experiment | Follow the advanced setup after the tests pass |
| Inspect implementation details | [File map](#file-map) and [method contract](research/METHOD.md) |

For a first Python/AI project, begin with [Laya or Needle](https://github.com/Runtime-weekly/runtime-tutorials/blob/main/START_HERE.md).
This repository is an advanced research example that evaluates generated Python.

## Requirements

- Python **3.12** on Linux. The exported tests were checked on Linux ARM64.
- Linux **Landlock** and **libseccomp.so.2** for child-process confinement.
- Disk space for the Python environment. No model is needed for the tests.
- For live runs only: a separately installed, locally running Ollama model.

Windows and macOS native execution are not supported by this sandbox. WSL,
containers and other Linux distributions have not been qualified here. A missing
sandbox stops execution; do not disable it to make a run proceed. Use a dedicated
environment without sensitive files; this is not a guarantee against every kernel
or native-library vulnerability.

## Install and test — no model required

```sh
git clone https://github.com/Runtime-weekly/dream-rsi-experiment.git
cd dream-rsi-experiment
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
```

The Python version should be 3.12.x. Run commands from the repository root.
Tests cover replay visibility, tie handling, checkpoint reuse, numerical correctness,
and denial of unrelated file access, writes, networking and process creation in
generated-code workers. They do not contact a model server or prove an AI improvement.

If a worker cannot start, inspect whether your Linux kernel supports Landlock
and your system provides libseccomp. Do not replace those checks with a timeout.
If package installation fails, inspect the platform/version mismatch before changing pins.

## Advanced: run a new experiment

First install and run your chosen model through [Ollama's official instructions](https://docs.ollama.com/).
This repository does not download models, start a server, or change a default model.
The recorded experiment used `qwen3.8:27b`; a different model is a new experiment
with separate hardware requirements and no identical-result promise.

```sh
curl http://127.0.0.1:11434/api/tags
```

In `configs/smoke.json`, set `model` to an exact installed model name and `endpoint`
to your loopback Ollama address. Keep this config and model unchanged within one run.
Only proceed when that server is yours to use and has capacity for the experiment.

```sh
python -m dreamrsi.cli run --config configs/smoke.json --out runs/first-run
python -m dreamrsi.cli status runs/first-run
```

The small configuration allows two outer rounds, two decisions per arm per round,
one worker and one policy revision per outer round: at most ten model requests.
Stopping early may reduce that count. This is a plumbing test, not the paper's budget.

Press Ctrl+C to request a pause after the current bounded operation. Resume with:

```sh
python -m dreamrsi.cli run --config configs/smoke.json --out runs/first-run --resume
```

Resume refuses changed model/source/configuration and uncertain unfinished requests.
Use a new output folder for a new experiment. Raw runs contain prompts, generated
code and local paths; they are ignored by Git and should not be posted without review.

## File map

| Path | Purpose |
| --- | --- |
| `dreamrsi/core.py` | Tree, prefix-only replay and policy selection |
| `dreamrsi/engine.py` | Online attempts, replay and comparison loop |
| `dreamrsi/model.py` | Loopback Ollama requests and saved-response reuse |
| `dreamrsi/task.py` | Independent Lasso correctness and timing evaluator |
| `dreamrsi/sandbox.py` | Worker filesystem and syscall restrictions |
| `tasks/lasso/baseline.py` | Starting numerical solver |
| `policies/fixed.py` | Fixed exploration baseline |
| `tests/` | CPU and mock-model tests |
| `RESULTS.md` | Historical result and public-export qualification |

## Sources and license

[Paper](https://arxiv.org/abs/2609.14858) · [Official project](https://www.dream-rsi.com/) ·
[Authors' repository](https://github.com/zhengkid/Dream-RSI)

RUNTIME's original implementation is [MIT licensed](LICENSE). The paper, authors'
code and third-party packages retain their own rights. No paper PDF, model weights,
private infrastructure, raw production logs or original repository history is bundled.
