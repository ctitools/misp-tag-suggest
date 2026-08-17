from __future__ import annotations

import subprocess
from pathlib import Path


def test_private_runtime_files_are_ignored():
    root = Path(__file__).resolve().parents[1]
    for name in (".env", "acl.json", "data/example.json", "artifacts/model.npz"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", name], cwd=root, check=False
        )
        assert result.returncode == 0, f"{name} is not ignored"


def test_no_generated_or_large_files_are_tracked():
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    assert not any("__pycache__" in path or path.endswith(".pyc") for path in tracked)
    assert not any((root / path).stat().st_size > 5_000_000 for path in tracked)
