"""Read-only official-release check. New code triggers review, never execution."""
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
from .core import save_json

REPO = "https://api.github.com/repos/zhengkid/Dream-RSI"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dream-rsi-reproduction-release-check",
                                              "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def classify(paths):
    # Evidence of source is a review signal, never a claim of a complete release.
    return sorted(p for p in paths if p.endswith((".py", ".sh", ".ipynb", ".toml", ".yaml", ".yml",
                                                  ".rs", ".go", ".cpp", ".c", ".js", ".ts"))
                  and not p.startswith(("assets/", ".github/")))


def check(path):
    path = Path(path)
    previous = json.loads(path.read_text()) if path.exists() else {}
    result = {"checked_at": datetime.now(timezone.utc).isoformat(),
              "repository": "https://github.com/zhengkid/Dream-RSI"}
    try:
        branch = get_json(REPO + "/branches/main")
        sha = branch["commit"]["sha"]
        tree = get_json(REPO + "/git/trees/" + sha + "?recursive=1")
        if tree.get("truncated"):
            raise ValueError("GitHub returned an incomplete tree")
        paths = sorted(n["path"] for n in tree["tree"] if n["type"] == "blob")
        source = classify(paths)
        releases = get_json(REPO + "/releases?per_page=5")
        tags = [r["tag_name"] for r in releases if not r["draft"]]
        result.update(commit=sha, files=paths, source_candidates=source, release_tags=tags,
                      changed=bool(previous.get("commit") and previous["commit"] != sha),
                      status="review_needed" if source or tags else "official_code_pending")
    except Exception as exc:
        result.update(status="check_failed", error=str(exc),
                      last_success=previous if previous.get("status") != "check_failed" else previous.get("last_success"))
    save_json(path, result)
    return result
