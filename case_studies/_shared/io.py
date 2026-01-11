from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

REPO_ROOT_MARKERS = {"requirements.txt", "Makefile", "case_studies"}

def find_repo_root(start: Path | None = None) -> Path:
    """Find repo root by walking up until we see expected marker files/folders."""
    p = (start or Path.cwd()).resolve()
    for _ in range(20):
        if all((p / m).exists() for m in REPO_ROOT_MARKERS):
            return p
        if p.parent == p:
            break
        p = p.parent
    return (start or Path.cwd()).resolve()

@dataclass(frozen=True)
class Paths:
    repo: Path
    data_raw: Path
    data_processed: Path
    artifacts_case: Path
    plots: Path
    reports: Path

def get_paths(case_name: str) -> Paths:
    repo = find_repo_root()
    data_raw = repo / "data" / "raw"
    data_processed = repo / "data" / "processed"
    artifacts_case = repo / "artifacts" / case_name
    plots = artifacts_case / "plots"
    reports = repo / "reports"

    for p in [data_raw, data_processed, artifacts_case, plots, reports]:
        p.mkdir(parents=True, exist_ok=True)

    return Paths(
        repo=repo,
        data_raw=data_raw,
        data_processed=data_processed,
        artifacts_case=artifacts_case,
        plots=plots,
        reports=reports,
    )

def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def safe_write_json(path: Path, obj: Dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
