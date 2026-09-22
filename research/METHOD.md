# Method contract and known differences

Primary references checked 2026-09-20:

- [Paper](https://arxiv.org/abs/2609.14858), supplied v1 PDF.
- [Official project](https://dream-rsi.com/).
- [Official repository](https://github.com/zhengkid/Dream-RSI).

## Reconstructed from section 3

1. One fixed discovery model and evaluator per experiment.
2. A root workspace and chains of attempts. Each non-root node has one parent;
   only the root or a currently observed leaf can be extended.
3. A policy chooses a batch of distinct eligible parents, bounded by worker count.
   Its source stays fixed throughout an online rollout.
4. Every attempt records source, proposal, parent, score, errors and costs. Online
   discovery reads prior completed histories plus the current observed tree.
5. Historical trees become replay worlds. A replay starts with the root only.
   Choosing root reveals its earliest unobserved recorded child. Choosing a leaf
   reveals its recorded next child, if one exists. Unrecorded results are never
   synthesized. A running policy receives only the revealed prefix.
6. Replay stops at an empty batch, its round cap, or complete history exhaustion.
7. Equation (1): best revealed score minus beta1 times represented generation
   count, plus beta2 times represented count divided by max(1, decision rounds).
8. A fixed LLM revises the policy code, using replay feedback. Each revision is
   evaluated on the same history pool. Incumbent is candidate zero and wins ties.
9. The best mean replay score selects the policy for the next online rollout.
   This guarantees only nondecreasing score on that fixed history pool.
10. Fixed-strategy comparison uses the same online allowance and discovery model.

The policy developer can inspect completed historical trees. The policy it writes
cannot inspect future nodes while a particular replay unfolds. Generated policy
code is isolated from those files; the trusted parent controls disclosure.

## Explicit reconstruction choices

- **Objective:** the main text's equation (1). Appendix B also describes a beta
  sweep/AUC/parallel-penalty evaluator; its complete implementation is unreleased.
  We do not silently mix that objective with equation (1).
- **Interface:** `choose(observation) -> list[parent_id]`, implementing section 3's
  action set. This is our interface, not an upstream-compatible claim. Appendix B
  names a different `OptimalPolicy.solve(question, budget)` API.
- **Root batches:** section 3 uses a set of distinct parents, so root appears once
  in a batch. New parallel branches therefore open across decision rounds. The
  Appendix B grid API can expose several root cells; this is a detail to reconcile
  against the official release.
- **Task:** Appendix A Problem 1's Lasso objective and +1e-6 reference-objective
  tolerance. Correctness uses fresh instances distinct from timing cases. All
  timed repetitions are also checked. Reference is tightly converged sklearn.
- **Instances:** three small synthetic regimes, eight lambdas, followed by three
  larger held-out regimes. This does not reconstruct SimpleTES's 17 cases or the
  six real-world downstream datasets used in the paper. No paper numbers are claimed.
- **Timing:** median of three warmed complete-path calls; trusted-parent wall
  clock includes pipe/result-serialization overhead. It excludes worker import,
  input loading and reference validation. Thus exact published runtimes are not
  comparable. Numerical evaluation is serialized to limit self-contention.
- **Models:** existing local Qwen via Ollama, with thinking disabled using the
  native top-level `think` parameter. The paper used Gemini CLI agents. We expose
  complete bounded history in the prompt and accept a single source file, rather
  than reproducing Gemini's interactive filesystem tool loop.
- **Budgets:** the shipped config is a small functional test. Published experiments
  used much larger budgets and worker counts. Total costs include policy-development
  calls so replay's lack of discovery calls is not mistaken for free policy writing.
- **Workspace:** one source-module task. Each node's source file is its workspace
  snapshot; the explicit primary parent determines inherited source. Failure code
  and diagnostics are retained for repair. Full multi-file workspace agents are not
  part of this first reconstruction.
- **Holdout:** evaluated once after selection for each arm; not fed back. Reliable
  comparative conclusions require repeated fresh experiments, not only replay.

## Acceptance for this checkpoint

- Prefix-only replay, legal batches and recorded child order tested.
- Incumbent retention and failed-policy rejection tested.
- Candidate output validated by a separate parent process.
- Generated code denied unrelated reads, writes, network and process creation.
- Full online/replay/revision/next-online path tested and real-model smoke attempted.
- Actual outcome and remaining gaps written separately from this method contract.

