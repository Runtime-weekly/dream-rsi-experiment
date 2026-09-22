from pathlib import Path
import pytest
from dreamrsi.process import Worker


def test_sandbox_denies_other_files_writes_network_and_processes(tmp_path):
    secret = tmp_path / "outside.txt"
    secret.write_text("test sentinel")
    source = tmp_path / "policy.py"
    source.write_text('''import os, socket
def choose(obs):
    denied = []
    for operation in [lambda: open(obs['outside']).read(),
                      lambda: open(obs['write'], 'w'),
                      lambda: socket.socket(), lambda: os.fork()]:
        try:
            operation()
            denied.append(False)
        except PermissionError:
            denied.append(True)
    return denied
''')
    with Worker(source, "policy") as worker:
        assert worker.choose({"outside": str(secret), "write": str(tmp_path / "created")}) == [True]*4
    assert not (tmp_path / "created").exists()


def test_hung_policy_is_terminated(tmp_path):
    source = tmp_path / "policy.py"
    source.write_text("def choose(obs):\n    while True: pass\n")
    with Worker(source, "policy", timeout=.5) as worker:
        with pytest.raises(TimeoutError):
            worker.choose({})
    assert worker.proc.poll() is not None

