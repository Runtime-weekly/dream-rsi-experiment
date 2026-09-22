"""Section 3's tree, prefix-only observations and equation (1). No model calls."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Node:
    id: str
    parent: str | None
    score: float
    valid: bool
    artifact: str = ""
    proposal: str = ""
    error: str | None = None
    depth: int = 0
    cost: dict = field(default_factory=dict)
    artifact_sha256: str = ""


class Tree:
    def __init__(self, nodes=None):
        self.nodes = []
        for node in nodes or [Node("root", None, 0.0, True)]:
            self.add(node)

    def add(self, node):
        if not math.isfinite(node.score) or node.score < 0:
            raise ValueError("Scores must be finite and nonnegative")
        if any(n.id == node.id for n in self.nodes):
            raise ValueError("Duplicate node")
        if self.nodes:
            if node.parent not in self.legal():
                raise ValueError("Only the root or a current leaf can be extended")
            if node.depth != self.get(node.parent).depth + 1:
                raise ValueError("Incorrect node depth")
        elif node.id != "root" or node.parent is not None:
            raise ValueError("The first node must be root")
        if not node.valid and node.score != 0:
            raise ValueError("Invalid candidates cannot earn a score")
        self.nodes.append(node)

    def get(self, node_id):
        return next(n for n in self.nodes if n.id == node_id)

    def legal(self):
        parents = {n.parent for n in self.nodes}
        return [n.id for n in self.nodes if n.id == "root" or n.id not in parents]

    def observe(self, workers, decision, limit):
        # Only supplied prefix nodes; no filesystem paths, future children or tree sizes.
        return {"nodes": [{k: v for k, v in asdict(n).items()
                           if k not in ("artifact", "artifact_sha256", "cost")} for n in self.nodes],
                "legal": self.legal(), "max_parallelism": workers,
                "decision": decision, "decision_limit": limit}

    def dump(self):
        return [asdict(n) for n in self.nodes]

    @classmethod
    def load(cls, data):
        nodes = [Node(**n) for n in data]
        for node in nodes:
            if node.artifact_sha256 and digest(Path(node.artifact).read_bytes()) != node.artifact_sha256:
                raise ValueError("Saved candidate artifact changed: " + node.id)
        return cls(nodes)


def validate_batch(batch, tree, workers):
    if not isinstance(batch, list) or any(not isinstance(a, str) for a in batch):
        raise ValueError("Policy must return a list of node IDs")
    if len(batch) > workers or len(set(batch)) != len(batch):
        raise ValueError("Batch exceeds worker count or contains duplicate actions")
    if not set(batch) <= set(tree.legal()):
        raise ValueError("Batch contains an illegal or unseen node")


def replay(tree, policy, workers, rounds, cost_weight, parallel_weight):
    """Reveals recorded continuations only. Policies never receive the full tree."""
    prefix = Tree([tree.get("root")])
    trajectory = []
    stop = "round_budget"
    if len(tree.nodes) == 1:
        return {"value": tree.get("root").score, "best_score": tree.get("root").score,
                "represented_calls": 0, "rounds": 0, "trajectory": [], "stop": "history_exhausted"}
    for decision in range(rounds):
        batch = policy.choose(prefix.observe(workers, decision, rounds))
        validate_batch(batch, prefix, workers)
        if not batch:
            stop = "policy_stop"
            break
        seen = {n.id for n in prefix.nodes}
        revealed = []
        for parent in batch:
            child = next((n for n in tree.nodes if n.parent == parent and n.id not in seen), None)
            if child is not None:
                prefix.add(child)
                revealed.append(child.id)
        trajectory.append({"actions": batch, "revealed": revealed})
        if len(prefix.nodes) == len(tree.nodes):
            stop = "history_exhausted"
            break
    count = len(prefix.nodes) - 1
    quality = max(n.score for n in prefix.nodes)
    value = quality - cost_weight * count + parallel_weight * count / max(1, len(trajectory))
    return {"value": value, "best_score": quality, "represented_calls": count,
            "rounds": len(trajectory), "trajectory": trajectory, "stop": stop}


def select_policy(evaluations):
    """Incumbent must be first. Stable max keeps it on a tie."""
    viable = [e for e in evaluations if e.get("valid")]
    if not viable or not evaluations[0].get("valid"):
        raise RuntimeError("Incumbent replay failed; do not promote a replacement")
    return max(viable, key=lambda e: e["mean_value"])
