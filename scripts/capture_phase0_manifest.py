from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


CONFIG_FLAGS = (
    "FINANCE_WEB_DEMO",
    "FINANCE_MAX_BLOCKS",
    "FINANCE_DEMO_KEYWORDS",
    "FINANCE_SKIP_IMAGE",
    "FINANCE_SKIP_GEN",
    "FINANCE_EMBED_BATCH_SIZE",
    "ENABLE_GRAPH",
    "FINANCE_RERANK_BACKEND",
    "FINANCE_GEN_BACKEND",
    "DEEPSEEK_MODEL",
    "HF_HOME",
    "HF_ENDPOINT",
)

SECRET_NAMES = ("DEEPSEEK_API_KEY", "FINANCE_API_KEY", "OPENAI_API_KEY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "missing"}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "status": "present",
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def git_head(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_status(root: Path) -> list[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str]:
    pairs = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            pairs[name.lower()] = distribution.version
    return dict(sorted(pairs.items()))


def build_manifest(
    finance_root: Path,
    agent_root: Path,
    kb_path: Path,
    test_set: Path,
    environ: Mapping[str, str],
) -> dict:
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "repositories": {
            "finance": {
                "path": str(finance_root.resolve()),
                "git_head": git_head(finance_root),
                "git_status": git_status(finance_root),
            },
            "agent": {
                "path": str(agent_root.resolve()),
                "git_head": git_head(agent_root),
                "git_status": git_status(agent_root),
            },
        },
        "artifacts": {
            "knowledge_base": file_record(kb_path),
            "test_set": file_record(test_set),
            "requirements": file_record(finance_root / "requirements.txt"),
            "model_registry": file_record(finance_root / "configs" / "model_registry.json"),
            "manifest_script": file_record(finance_root / "scripts" / "capture_phase0_manifest.py"),
            "benchmark_script": file_record(finance_root / "scripts" / "benchmark_regression.py"),
            "parser_audit_script": file_record(finance_root / "scripts" / "audit_parser_baseline.py"),
            "agent_graph": file_record(agent_root / "graph" / "graph.py"),
            "agent_web_search": file_record(agent_root / "graph" / "nodes" / "web_search.py"),
            "agent_pyproject": file_record(agent_root / "pyproject.toml"),
        },
        "environment_flags": {
            **{name: environ.get(name) for name in CONFIG_FLAGS},
            **{f"{name}_configured": bool(environ.get(name)) for name in SECRET_NAMES},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture phase-0 reproducibility manifest")
    parser.add_argument("--finance-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--kb-path", type=Path, default=Path("output/knowledge_base.json"))
    parser.add_argument("--test-set", type=Path, default=Path("test/merged.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("output/baseline/manifest.json"))
    args = parser.parse_args()

    manifest = build_manifest(
        finance_root=args.finance_root.resolve(),
        agent_root=args.agent_root.resolve(),
        kb_path=args.kb_path.resolve(),
        test_set=args.test_set.resolve(),
        environ=os.environ,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
