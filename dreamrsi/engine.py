"""Bounded online -> replay -> policy revision -> online reconstruction."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
from .core import Node, Tree, digest, replay, save_json, select_policy, validate_batch
from .process import Worker
from .task import SPEC, evaluate

ROOT = Path(__file__).resolve().parents[1]


def history_text(trees):
    # Full recorded proposals and evaluation evidence, including failures.
    return json.dumps([[{k: v for k, v in n.items() if k != "artifact"} for n in t.dump()]
                       for t in trees], allow_nan=False)


def failure_sources(trees):
    return json.dumps([{ "id": n.id, "error": n.error, "source": Path(n.artifact).read_text()}
                       for tree in trees for n in tree.nodes if not n.valid and n.artifact])


def source_hashes():
    paths = sorted(list((ROOT / "dreamrsi").glob("*.py")) + list((ROOT / "policies").glob("*.py"))
                   + list((ROOT / "tasks/lasso").glob("*.py")))
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}


class Experiment:
    def __init__(self, directory, config, model, stop=lambda: False):
        self.directory, self.config, self.model, self.stop = Path(directory), config, model, stop
        self.directory.mkdir(parents=True, exist_ok=True)

    def attempt(self, tree, parent_id, node_id, target, history, seed):
        parent = tree.get(parent_id)
        parent_source = Path(parent.artifact).read_text()
        prompt = (SPEC + "\nStart from this parent workspace source:\n```python\n" + parent_source +
                  "\n```\nRead all prior proposals and measured results below before proposing a change. "
                  "Distinguish a failed idea from a repairable bug. Avoid repeating saturated mechanisms.\n" +
                  history_text(history + [tree]) + "\nSource for failed attempts, for concrete debugging:\n" +
                  failure_sources(history + [tree]))
        cost = {"model_calls": 0}
        source = ""
        proposal = ""
        result = {"valid": False, "score": 0.0}
        try:
            source, proposal, cost = self.model.generate(prompt, target, "solve_path", seed)
            (target / "candidate.py").write_text(source)
            result = evaluate(target / "candidate.py", seed)
        except Exception as exc:
            result["error"] = type(exc).__name__ + ": " + str(exc)
            cost["model_calls"] = int((target / "request.json").exists())
        target.mkdir(parents=True, exist_ok=True)
        # A failed generation inherits parent workspace, with its failure recorded.
        if not source:
            (target / "candidate.py").write_text(parent_source)
        (target / "proposal.md").write_text(proposal or result.get("error", "No proposal"))
        save_json(target / "evaluation.json", result)
        node = Node(node_id, parent_id, float(result["score"]), result["valid"],
                    str(target / "candidate.py"), proposal, result.get("error"), parent.depth + 1,
                    {**cost, "evaluation_seconds": result.get("evaluation_seconds", 0)},
                    digest((target / "candidate.py").read_bytes()))
        save_json(target / "node.json", node.__dict__)
        return node

    def online(self, target, policy_path, history, round_index, baseline):
        tree_path, state_path = target / "tree.json", target / "online.json"
        target.mkdir(parents=True, exist_ok=True)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"decisions": [], "complete": False}
        tree = Tree.load(state["tree"]) if "tree" in state else Tree([
            Node("root", None, baseline["score"], True, str(ROOT / "tasks/lasso/baseline.py"), "Fixed initial solver",
                 artifact_sha256=digest((ROOT / "tasks/lasso/baseline.py").read_bytes()))])
        if state["complete"]:
            return tree
        # Policy state is reset on startup. Replaying prior observations reconstructs it.
        with Worker(policy_path, "policy") as policy:
            for prior in state["decisions"]:
                if policy.choose(prior["observation"]) != prior["actions"]:
                    raise RuntimeError("Policy is nondeterministic; cannot resume")
            for decision in range(len(state["decisions"]), self.config["online_rounds"]):
                if self.stop():
                    raise InterruptedError("Paused at a checkpoint")
                observation = tree.observe(self.config["workers"], decision, self.config["online_rounds"])
                batch = policy.choose(observation)
                validate_batch(batch, tree, self.config["workers"])
                if not batch:
                    state["stop"] = "policy_stop"
                    break
                assignments = [(p, "n%05d" % (len(tree.nodes) + i)) for i, p in enumerate(batch)]
                save_json(target / "pending.json", {"decision": decision, "assignments": assignments})
                def execute(pair):
                    parent, node_id = pair
                    location = target / "attempts" / node_id
                    if (location / "node.json").exists():
                        node = Node(**json.loads((location / "node.json").read_text()))
                        if node.artifact_sha256 and digest(Path(node.artifact).read_bytes()) != node.artifact_sha256:
                            raise ValueError("Interrupted batch artifact changed: " + node.id)
                        return node
                    # An abrupt interruption after sending a request is not silently retried.
                    if (location / "request.json").exists() and not (location / "response.json").exists():
                        raise RuntimeError("Unconfirmed model call: " + str(location))
                    return self.attempt(tree, parent, node_id, location, history,
                                        self.config["seed"] + round_index * 1000 + int(node_id[1:]))
                with ThreadPoolExecutor(max_workers=self.config["workers"]) as pool:
                    children = list(pool.map(execute, assignments))
                for node in children:
                    tree.add(node)
                state["decisions"].append({"observation": observation, "actions": batch})
                state["tree"] = tree.dump()
                # This single atomic checkpoint is authoritative; tree.json is an export.
                save_json(state_path, state)
                save_json(tree_path, tree.dump())
                (target / "pending.json").unlink(missing_ok=True)
                print(f"{target.parent.name}/{target.name}: decision {decision+1}, "
                      f"{len(tree.nodes)-1} attempts, best {max(n.score for n in tree.nodes):.5f}", flush=True)
        state.update(complete=True, stop=state.get("stop", "round_budget"))
        state["tree"] = tree.dump()
        save_json(tree_path, tree.dump())
        save_json(state_path, state)
        return tree

    def replay_policy(self, path, trees):
        outcomes = []
        try:
            for tree in trees:
                with Worker(path, "policy") as policy:
                    outcomes.append(replay(tree, policy, self.config["workers"],
                                           self.config["replay_rounds"], self.config["cost_weight"],
                                           self.config["parallel_weight"]))
            return {"path": str(path), "sha256": digest(Path(path).read_bytes()), "valid": True,
                    "mean_value": sum(r["value"] for r in outcomes) / len(outcomes), "worlds": outcomes}
        except Exception as exc:
            return {"path": str(path), "valid": False, "error": str(exc), "worlds": outcomes}

    def dream(self, target, incumbent, trees, round_index):
        target.mkdir(parents=True, exist_ok=True)
        receipt = target / "selection.json"
        if receipt.exists():
            selected = json.loads(receipt.read_text())["selected"]
            path = Path(selected["path"])
            if digest(path.read_bytes()) != selected["sha256"]:
                raise ValueError("Saved selected policy changed")
            return path
        evaluations = [self.replay_policy(incumbent, trees)]
        working = incumbent
        for revision in range(1, self.config["policy_versions"]):
            if self.stop():
                raise InterruptedError("Paused before policy revision")
            location = target / f"revision-{revision}"
            prompt = ("Improve ONLY the exploration policy, not the Lasso solver or evaluator.\n"
                      "Return complete Python source defining choose(observation), returning a list of distinct legal node IDs. "
                      "An empty list stops the rollout. State may persist in module globals within one rollout; "
                      "every replay world starts a new process. Use deterministic decisions.\n"
                      "observation has nodes (revealed prefix only, each id,parent,score,valid,error,proposal,depth), "
                      "legal (root plus current leaves), max_parallelism, decision, decision_limit. "
                      "The root opens one new independent branch. A leaf continues that branch. "
                      "Return at most max_parallelism distinct actions; only actions legal before this call. "
                      "Do not invent or inspect future outcomes or filesystem history. Compare whole branch trajectories, "
                      "weigh recovery of implementation bugs against opening new roots, and stop only after considering "
                      "remaining alternatives. Do not close a branch just because one attempt failed.\n"
                      "Section 3 equation (1) score per replay = best revealed score - cost_weight * revealed_attempts "
                      "+ parallel_weight * revealed_attempts/max(1, decision_rounds). "
                      "Selection uses mean score across all fixed historical trees.\n"
                      f"cost_weight={self.config['cost_weight']}; parallel_weight={self.config['parallel_weight']}\n"
                      "Current revision:\n```python\n" + Path(working).read_text() + "\n```\n"
                      "Earlier revisions and replay feedback:\n" + json.dumps(evaluations) + "\n"
                      "Completed histories available to you as the policy developer:\n" + history_text(trees))
            location.mkdir(parents=True, exist_ok=True)
            path = location / "policy.py"
            try:
                source, proposal, cost = self.model.generate(prompt, location, "choose",
                                                             self.config["seed"] + 70000 + round_index * 100 + revision)
                path.write_text(source)
                working = path
                outcome = self.replay_policy(path, trees)
                outcome.update(proposal=proposal, cost=cost)
            except Exception as exc:
                outcome = {"path": str(path), "valid": False, "error": str(exc)}
            evaluations.append(outcome)
            save_json(location / "replay.json", outcome)
        selected = select_policy(evaluations)
        save_json(receipt, {"incumbent": str(incumbent), "evaluations": evaluations, "selected": selected,
                           "guarantee": "Nondecreasing mean replay score on these historical trees only"})
        print(f"{target}: selected {selected['path']} at replay score {selected['mean_value']:.5f}", flush=True)
        return Path(selected["path"])

    def run(self):
        started = time.monotonic()
        base_path = self.directory / "baseline.json"
        if base_path.exists():
            baseline = json.loads(base_path.read_text())
        else:
            baseline = evaluate(ROOT / "tasks/lasso/baseline.py", self.config["seed"])
            save_json(base_path, baseline)
        if not baseline["valid"]:
            raise RuntimeError("Initial solver failed: " + str(baseline))
        trees = {"adaptive": [], "fixed": []}
        policies = {arm: ROOT / "policies/fixed.py" for arm in trees}
        # Both arms run each round, with the adaptive arm first (not counterbalanced).
        for round_index in range(self.config["rounds"]):
            for arm in ("adaptive", "fixed"):
                location = self.directory / arm / f"round-{round_index+1}"
                tree = self.online(location, policies[arm], trees[arm], round_index, baseline)
                trees[arm].append(tree)
                if arm == "adaptive":
                    policies[arm] = self.dream(location / "dream", policies[arm], trees[arm], round_index)
        report = {"status": "completed", "model": self.model.identity, "config": self.config,
                  "qualification": "Local method reconstruction; not published benchmark replication",
                  "arms": {}}
        for arm, history in trees.items():
            best = max((n for t in history for n in t.nodes), key=lambda n: n.score)
            heldout_path = self.directory / arm / "heldout.json"
            if heldout_path.exists():
                heldout = json.loads(heldout_path.read_text())
            else:
                heldout = evaluate(best.artifact, self.config["seed"] + 900000, heldout=True)
                save_json(heldout_path, heldout)
            records = [json.loads(p.read_text()) for p in (self.directory / arm).rglob("response.json")]
            requests = list((self.directory / arm).rglob("request.json"))
            discovery_calls = sum("attempts" in p.relative_to(self.directory / arm).parts for p in requests)
            online = [n for t in history for n in t.nodes if n.id != "root"]
            report["arms"][arm] = {"best_search_score": best.score, "best_artifact": best.artifact,
                "heldout": heldout, "discovery_calls": discovery_calls, "all_model_calls": len(requests),
                "policy_development_calls": len(requests) - discovery_calls,
                "candidate_attempts": len(online),
                "unique_candidate_sources": len({digest(Path(n.artifact).read_bytes()) for n in online}),
                "best_artifact_sha256": digest(Path(best.artifact).read_bytes()),
                "valid_candidates": sum(n.valid for n in online),
                "model_seconds": sum(r.get("seconds", 0) for r in records),
                "output_tokens": sum(r.get("answer", {}).get("eval_count", 0) for r in records),
                "selected_policy": str(policies[arm])}
        report["invocation_seconds"] = time.monotonic() - started
        report["conclusion"] = "A small smoke run verifies execution only; it does not establish a Dream-RSI advantage."
        save_json(self.directory / "report.json", report)
        return report
