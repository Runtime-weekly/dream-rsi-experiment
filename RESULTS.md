# What the experiment showed

Recorded model run: **September 20, 2026**. Public-export review: **September 21, 2026**.
[Watch the episode](https://youtu.be/opBzkTGTx8E).

## The result in plain language

We tested a small independently written loop with a fixed local Qwen model.
It generated numerical solver attempts, recorded their outcomes, proposed two
search-policy revisions, and replayed them against the saved history.

Neither revision won. Both tied the current policy, so the original stayed in use.
That is a completed experiment, but it is not evidence of improved exploration,
model-weight training, or a general self-improving assistant.

| Recorded measurement | Result |
| --- | --- |
| Total model requests | 10: eight solver attempts and two policy revisions |
| Solver attempts passing the numerical checks | 8 / 8 |
| Round 1 replay: original / proposed | 0.4365368797 / 0.4365368797 |
| Round 2 replay: original / proposed | 0.4628173206 / 0.4628173206 |
| Policy selection | Original retained in both rounds |
| Final correctness on new cases | Three cases passed for each arm |

Each replay row compares candidates against the same history. The different scores
between rows reflect different histories, not a policy improvement.

## What we cannot conclude

The two arms used the same original search policy. Their small, noisy timing
differences cannot show that policy improvement helped. Passing a few synthetic
numerical checks is not a general accuracy or safety claim. The task, budgets,
model and execution interface differ from those in the paper.

The underlying experiment retains the best score on its recorded replay histories.
That does not guarantee better results on future, unseen work.

## Which code is shared here

This standalone export contains the recorded experiment's implementation plus its
subsequent checkpoint-integrity and edge-case fixes. The historical ten-request run
was not repeated after those fixes. Public-export verification reruns the CPU/mock
tests, not a new live-model benchmark. See [METHOD.md](research/METHOD.md) for the
specific reconstruction choices.

The numbers above are a manually reviewed summary of retained local evidence.
Raw requests, full run trees and private host paths are not distributed here.
You can inspect and test the implementation, then generate your own results.

Public-export check: **24 tests passed** in a fresh Python 3.12 environment on
Linux ARM64, using the pinned dependencies. These were CPU/mock tests, not a new
model-backed experiment.
