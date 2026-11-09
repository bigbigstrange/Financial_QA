# 阶段 0：冻结现状与建立基线实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不升级模型、不更换向量库、不重构正式业务架构的前提下，建立两个项目可复现、可比较、不会把模拟搜索当作证据的阶段 0 基线。

**Architecture:** `finance-qa-platform` 继续作为主仓库，新增纯标准库的环境快照、金融问答基线和解析质量审计；Agentic RAG 只增加可注入的图构建入口、离线分支测试，并将伪联网搜索改为失败关闭。所有运行产物写入已被 Git 忽略的 `output/baseline/`，代码和运行说明进入版本控制。

**Tech Stack:** Python 3.11、现有 FastAPI/RAG 依赖、LangGraph、标准库 `unittest`、JSON、PowerShell；本阶段不增加 Qdrant、Docling、Celery、Redis、PostgreSQL 或新模型。

**Spec:** `docs/superpowers/specs/2026-09-19-financial-agentic-rag-integration-design.md`

## Global Constraints

- 主仓库是 `C:\code\个人资料\code\Agentic_RAG\finance-qa-platform`。
- Agent 参考项目是 `C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag`。
- Python 目标版本为 3.11；不要复用指向失效 Python 路径的旧 `.venv`。
- 本阶段只修改基线脚本、测试、运行说明以及 Agentic RAG 的图构建和伪搜索节点。
- 不修改金融问答线上回答逻辑、Embedding、重排、生成模型、切块算法和数据库结构。
- 不将 `.env`、API Key、完整文档正文、真实用户信息或生成的 `output/baseline/` 产物提交到 Git。
- 不再把普通 LLM 生成文本命名为网络搜索结果或作为 Evidence。
- 每次基线运行必须记录代码版本、数据哈希、模型配置、功能开关和依赖版本。
- “有回答且有证据”字段命名为 `evidence_response_rate`，不得称为准确率或成功率。
- Git 提交前由用户配置真实 `user.name` 和 `user.email`；执行者不得编造身份。

## Review Focus

1. 测试集为空、缺少 `question` 或缺少 `question_id` 时，基线脚本必须清楚失败，不能生成 100% 指标；Task 2 的单元测试覆盖。
2. 知识库缺失、JSON 损坏、出现重复 `block_id` 或无内容文档时，解析审计必须返回非零退出码并写明原因；Task 3 的单元测试覆盖。
3. 外部问题路由到 `websearch` 时，系统必须失败关闭且不得调用 DeepSeek 伪造来源；Task 4 的单元测试覆盖。
4. 生成结果第一次无依据、第二次有依据时，图必须真实执行第二次生成，防止无条件 `GENERATE -> END` 提前结束；Task 4 的分支测试覆盖。
5. 环境快照中即使存在 `DEEPSEEK_API_KEY` 等变量，也只能记录“是否配置”，不得写入秘密值；Task 1 的单元测试覆盖。

---

## 文件变更总览

### 金融 RAG 主仓库

| 文件 | 操作 | 职责 |
|---|---|---|
| `tests/__init__.py` | 新建 | 确保项目测试包稳定导入 |
| `scripts/capture_phase0_manifest.py` | 新建 | 生成代码、数据、配置和运行环境快照 |
| `tests/test_capture_phase0_manifest.py` | 新建 | 验证哈希稳定、秘密不泄露、缺失文件可表示 |
| `scripts/benchmark_regression.py` | 修改 | 输出逐案例答案、证据、错误和正确命名的基线指标 |
| `tests/test_benchmark_regression.py` | 新建 | 用 FakeQA 离线验证评测聚合逻辑 |
| `scripts/audit_parser_baseline.py` | 新建 | 审计现有 DocIR/knowledge_base/parser_report 的结构质量 |
| `tests/test_audit_parser_baseline.py` | 新建 | 验证空文档、重复 ID、坏 JSON 和合法知识库 |
| `docs/baseline/phase-0-runbook.md` | 新建 | 固化 smoke/full 命令、产物和故障处理 |
| `README.md` | 修改 | 增加阶段 0 基线入口链接 |

### Agentic RAG 参考项目

| 文件 | 操作 | 职责 |
|---|---|---|
| `graph/graph.py` | 修改 | 提供 `build_app()`，删除无条件生成结束边，便于离线测试 |
| `graph/nodes/web_search.py` | 修改 | 删除伪搜索模型调用，改为明确失败关闭 |
| `graph/tests/__init__.py` | 新建 | 测试包标记 |
| `graph/tests/test_graph_routes.py` | 新建 | 覆盖向量检索、重试、搜索失败关闭和循环上限 |

---

### Task 1: 建立可复现环境与数据快照

**Files:**

- Create: `scripts/capture_phase0_manifest.py`
- Create: `tests/__init__.py`
- Create: `tests/test_capture_phase0_manifest.py`

**Interfaces:**

- Produces: `sha256_file(path: Path) -> str`
- Produces: `build_manifest(finance_root: Path, agent_root: Path, kb_path: Path, test_set: Path, environ: Mapping[str, str]) -> dict`
- Produces: CLI 输出 `output/baseline/manifest.json`
- Consumed by: Task 2 的基线报告和 Task 5 的运行手册

- [ ] **Step 1: 编写秘密不泄露和哈希稳定的失败测试**

Create empty `tests/__init__.py`, then create `tests/test_capture_phase0_manifest.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from scripts.capture_phase0_manifest import build_manifest, sha256_file


class Phase0ManifestTests(unittest.TestCase):
    def test_hash_is_stable_and_missing_file_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "payload.txt"
            payload.write_text("baseline", encoding="utf-8")

            self.assertEqual(sha256_file(payload), sha256_file(payload))
            manifest = build_manifest(
                finance_root=root,
                agent_root=root / "missing-agent",
                kb_path=payload,
                test_set=root / "missing.xlsx",
                environ={},
            )

            self.assertEqual(manifest["artifacts"]["knowledge_base"]["status"], "present")
            self.assertEqual(manifest["artifacts"]["test_set"]["status"], "missing")

    def test_secret_values_never_appear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret = "sk-do-not-write-this"
            manifest = build_manifest(
                finance_root=root,
                agent_root=root,
                kb_path=root / "kb.json",
                test_set=root / "cases.xlsx",
                environ={"DEEPSEEK_API_KEY": secret, "FINANCE_WEB_DEMO": "1"},
            )

            serialized = json.dumps(manifest, ensure_ascii=False)
            self.assertNotIn(secret, serialized)
            self.assertTrue(manifest["environment_flags"]["DEEPSEEK_API_KEY_configured"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
python -m unittest tests.test_capture_phase0_manifest -v
```

Expected: `ModuleNotFoundError: No module named 'scripts.capture_phase0_manifest'`。

- [ ] **Step 3: 实现最小快照脚本**

Create `scripts/capture_phase0_manifest.py` with these exact public functions and fields:

```python
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
    "FINANCE_SKIP_IMAGE",
    "ENABLE_GRAPH",
    "FINANCE_RERANK_BACKEND",
    "FINANCE_GEN_BACKEND",
    "DEEPSEEK_MODEL",
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
            "agent_graph": file_record(agent_root / "graph" / "graph.py"),
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
```

- [ ] **Step 4: 运行测试并验证通过**

Run:

```powershell
python -m unittest tests.test_capture_phase0_manifest -v
```

Expected: 2 tests run, 0 failures, 0 errors。

- [ ] **Step 5: 生成本机阶段 0 快照并确认秘密未写入**

Run:

```powershell
$AgentRoot = 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
python scripts/capture_phase0_manifest.py --agent-root $AgentRoot
Select-String -LiteralPath 'output/baseline/manifest.json' -Pattern 'sk-' -SimpleMatch
```

Expected: manifest 路径被打印；`Select-String` 无输出。缺失的知识库或测试集必须记录 `status: missing`，不得使快照脚本崩溃。

- [ ] **Step 6: 提交 Task 1**

```powershell
git add scripts/capture_phase0_manifest.py tests/__init__.py tests/test_capture_phase0_manifest.py
git commit -m "test: capture phase zero environment manifest"
```

Expected: 只提交上述两个文件。若 Git 身份未配置，停止提交并向用户索取真实身份信息。

---

### Task 2: 修正金融问答基线指标并保留逐案例证据

**Files:**

- Modify: `scripts/benchmark_regression.py:23-93`
- Create: `tests/test_benchmark_regression.py`

**Interfaces:**

- Consumes: `FinancialQASystem.answer_question(question: str, question_id: str) -> dict`
- Consumes: Task 1 的 `sha256_file()`
- Produces: `run_cases(qa, cases: list[dict], clock: Callable[[], float], rss_bytes: Callable[[], int] | None) -> dict`
- Produces: `output/baseline/qa_report.json`

- [ ] **Step 1: 编写空测试集、缺列、正常和异常案例的失败测试**

Create `tests/test_benchmark_regression.py`:

```python
import unittest

from scripts.benchmark_regression import run_cases


class FakeQA:
    def answer_question(self, question, question_id):
        if question == "boom":
            raise RuntimeError("offline failure")
        return {
            "answer": "基于证据的答案",
            "confidence": 0.8,
            "evidence": [
                {
                    "block_id": "doc-v1-b1",
                    "source_doc": "demo.pdf",
                    "page": 2,
                    "score": 0.7,
                    "text": "原始证据正文",
                }
            ],
        }


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class BenchmarkRegressionTests(unittest.TestCase):
    def test_empty_cases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "test set is empty"):
            run_cases(FakeQA(), [], clock=SequenceClock([]), rss_bytes=SequenceClock([]))

    def test_required_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "question_id and question"):
            run_cases(
                FakeQA(),
                [{"question": "q"}],
                clock=SequenceClock([]),
                rss_bytes=SequenceClock([]),
            )

    def test_report_preserves_answer_and_evidence_without_calling_it_accuracy(self):
        report = run_cases(
            FakeQA(),
            [{"question_id": "1", "question": "q"}],
            clock=SequenceClock([1.0, 1.25]),
            rss_bytes=SequenceClock([100, 150]),
        )
        self.assertEqual(report["metrics"]["evidence_response_rate"], 1.0)
        self.assertNotIn("success_rate", report["metrics"])
        self.assertEqual(report["cases"][0]["answer"], "基于证据的答案")
        self.assertEqual(report["cases"][0]["evidence"][0]["block_id"], "doc-v1-b1")
        self.assertEqual(report["metrics"]["max_observed_rss_bytes"], 150)

    def test_one_case_failure_is_recorded_and_run_continues(self):
        report = run_cases(
            FakeQA(),
            [
                {"question_id": "1", "question": "boom"},
                {"question_id": "2", "question": "q"},
            ],
            clock=SequenceClock([1.0, 1.1, 2.0, 2.2]),
            rss_bytes=SequenceClock([100, 120, 120, 180]),
        )
        self.assertEqual(report["metrics"]["error_count"], 1)
        self.assertEqual(report["cases"][0]["error_type"], "RuntimeError")
        self.assertEqual(report["cases"][1]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认缺少 `run_cases` 而失败**

Run:

```powershell
python -m unittest tests.test_benchmark_regression -v
```

Expected: ImportError 指出 `run_cases` 不存在。

- [ ] **Step 3: 在原脚本中实现纯聚合函数**

Add these functions above `main()` in `scripts/benchmark_regression.py`:

```python
from statistics import mean
from typing import Callable, Iterable


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return round(ordered[index], 4)


def _slim_evidence(items: list[dict]) -> list[dict]:
    fields = ("block_id", "source_doc", "page", "score", "fusion_score", "text")
    return [{key: item.get(key) for key in fields if key in item} for item in items]


def _process_rss_bytes() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def run_cases(
    qa,
    cases: Iterable[dict],
    clock: Callable[[], float] = time.perf_counter,
    rss_bytes: Callable[[], int] | None = None,
) -> dict:
    rows = list(cases)
    if not rows:
        raise ValueError("test set is empty")
    if any("question_id" not in row or "question" not in row for row in rows):
        raise ValueError("each test case requires question_id and question")

    details = []
    latencies = []
    rss_observations = []
    evidence_responses = 0
    errors = 0
    rss_bytes = rss_bytes or _process_rss_bytes
    for row in rows:
        qid = str(row["question_id"])
        question = str(row["question"])
        started = clock()
        rss_before = rss_bytes()
        try:
            result = qa.answer_question(question, qid)
            latency = round(clock() - started, 4)
            rss_after = rss_bytes()
            evidence = _slim_evidence(result.get("evidence") or [])
            answered = bool((result.get("answer") or "").strip()) and bool(evidence)
            evidence_responses += int(answered)
            details.append(
                {
                    "question_id": qid,
                    "question": question,
                    "status": "completed",
                    "latency_sec": latency,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": rss_after,
                    "rss_delta_bytes": rss_after - rss_before,
                    "answer": result.get("answer") or "",
                    "confidence": result.get("confidence"),
                    "evidence": evidence,
                    "responded_with_evidence": answered,
                    "error_type": None,
                    "error": None,
                }
            )
        except Exception as exc:
            latency = round(clock() - started, 4)
            rss_after = rss_bytes()
            errors += 1
            details.append(
                {
                    "question_id": qid,
                    "question": question,
                    "status": "error",
                    "latency_sec": latency,
                    "rss_before_bytes": rss_before,
                    "rss_after_bytes": rss_after,
                    "rss_delta_bytes": rss_after - rss_before,
                    "answer": "",
                    "confidence": None,
                    "evidence": [],
                    "responded_with_evidence": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        latencies.append(latency)
        rss_observations.extend((rss_before, rss_after))

    return {
        "schema_version": 1,
        "metrics": {
            "sample_count": len(details),
            "error_count": errors,
            "evidence_response_rate": round(evidence_responses / len(details), 4),
            "avg_latency_sec": round(mean(latencies), 4),
            "p50_latency_sec": _percentile(latencies, 0.50),
            "p95_latency_sec": _percentile(latencies, 0.95),
            "max_observed_rss_bytes": max(rss_observations),
        },
        "cases": details,
    }
```

- [ ] **Step 4: 让 CLI 调用 `run_cases` 并绑定输入快照**

Replace the current manual loop in `main()` with:

```python
    required = {"question_id", "question"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"test set missing columns: {missing}")

    report = run_cases(qa, df[["question_id", "question"]].to_dict("records"))
    report["inputs"] = {
        "knowledge_base": {
            "path": str(Path(args.kb_path).resolve()),
            "sha256": sha256_file(Path(args.kb_path)),
        },
        "test_set": {
            "path": str(Path(args.test_set).resolve()),
            "sha256": sha256_file(Path(args.test_set)),
        },
        "image_backend": args.image_backend,
        "max_samples": args.max_samples,
        "resolved_models": {
            "embedding": getattr(qa, "embed_model", None),
            "image_backend": getattr(qa, "image_backend", None),
            "generation": getattr(qa, "_gen_model", None),
        },
    }
```

Import `sha256_file` from `scripts.capture_phase0_manifest`. Change defaults to:

```python
    parser.add_argument("--output", default="./output/baseline/qa_report.json")
    parser.add_argument("--final-output", default="./output/baseline/final_report.json")
```

Update the final summary keys to read from `report["metrics"]` and use `evidence_response_rate`. Remove all `success_rate` and `qa_success_rate` labels from generated reports.

- [ ] **Step 5: 运行离线测试并验证通过**

Run:

```powershell
python -m unittest tests.test_benchmark_regression -v
```

Expected: 4 tests run, 0 failures, 0 errors。

- [ ] **Step 6: 使用 3 条真实问题执行 smoke 基线**

Run:

```powershell
python scripts/benchmark_regression.py `
  --kb-path output/knowledge_base.json `
  --test-set test/merged.xlsx `
  --output output/baseline/qa_report.json `
  --final-output output/baseline/final_report.json `
  --max-samples 3
```

Expected: 脚本以退出码 0 结束，两个报告存在；每个案例包含 `answer`、`evidence`、`status`、`latency_sec`。如果模型或知识库不可用，逐案例记录错误；不得生成虚假的正确率。

- [ ] **Step 7: 提交 Task 2**

```powershell
git add scripts/benchmark_regression.py tests/test_benchmark_regression.py
git commit -m "test: capture finance QA baseline evidence"
```

---

### Task 3: 建立文档解析质量基线

**Files:**

- Create: `scripts/audit_parser_baseline.py`
- Create: `tests/test_audit_parser_baseline.py`

**Interfaces:**

- Consumes: `knowledge_base.json` 与 `parser_report.json`
- Produces: `audit_knowledge_base(kb: dict, parser_report: dict) -> dict`
- Produces: CLI 输出 `output/baseline/parser_audit.json`

- [ ] **Step 1: 编写合法知识库、重复 ID、空文档和坏 JSON 的失败测试**

Create `tests/test_audit_parser_baseline.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_parser_baseline import audit_knowledge_base, load_json


def valid_kb():
    return {
        "documents": [
            {
                "doc_id": "doc-1",
                "pages": [
                    {
                        "page_no": 1,
                        "blocks": [
                            {"block_id": "doc-1-p1-b1", "block_type": "paragraph", "text": "正文"},
                            {"block_id": "doc-1-p1-t1", "block_type": "table", "text": "", "table_md": "|A|"},
                        ],
                    }
                ],
            }
        ]
    }


class ParserBaselineTests(unittest.TestCase):
    def test_valid_kb_is_clean(self):
        report = audit_knowledge_base(valid_kb(), {"summary": {"failed_files": 0}})
        self.assertTrue(report["passed"])
        self.assertEqual(report["counts"]["blocks"], 2)
        self.assertEqual(report["counts"]["table_blocks"], 1)

    def test_duplicate_block_id_fails(self):
        kb = valid_kb()
        kb["documents"][0]["pages"][0]["blocks"].append(
            {"block_id": "doc-1-p1-b1", "block_type": "paragraph", "text": "重复"}
        )
        report = audit_knowledge_base(kb, {"summary": {"failed_files": 0}})
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"]["duplicate_block_ids"], ["doc-1-p1-b1"])

    def test_document_without_blocks_fails(self):
        report = audit_knowledge_base(
            {"documents": [{"doc_id": "empty", "pages": []}]},
            {"summary": {"failed_files": 0}},
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["issues"]["documents_without_blocks"], ["empty"])

    def test_bad_json_has_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_json(path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认模块不存在**

```powershell
python -m unittest tests.test_audit_parser_baseline -v
```

Expected: ModuleNotFoundError。

- [ ] **Step 3: 实现解析审计脚本**

Create `scripts/audit_parser_baseline.py`:

```python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def audit_knowledge_base(kb: dict, parser_report: dict) -> dict:
    documents = kb.get("documents") or []
    block_ids = []
    block_types = Counter()
    documents_without_blocks = []
    empty_content_blocks = []
    page_count = 0
    block_count = 0
    ocr_blocks = 0

    for document in documents:
        doc_id = str(document.get("doc_id") or document.get("source_path") or "unknown")
        document_block_count = 0
        for page in document.get("pages") or []:
            page_count += 1
            for block in page.get("blocks") or []:
                block_count += 1
                document_block_count += 1
                block_id = str(block.get("block_id") or "")
                if block_id:
                    block_ids.append(block_id)
                block_type = str(block.get("block_type") or "unknown")
                block_types[block_type] += 1
                has_content = bool((block.get("text") or "").strip()) or bool(
                    (block.get("table_md") or "").strip()
                )
                if not has_content:
                    empty_content_blocks.append(block_id or f"{doc_id}:missing-id")
                if (block.get("meta") or {}).get("ocr_fallback"):
                    ocr_blocks += 1
        if document_block_count == 0:
            documents_without_blocks.append(doc_id)

    counts = Counter(block_ids)
    duplicate_block_ids = sorted(key for key, count in counts.items() if count > 1)
    parser_failures = int((parser_report.get("summary") or {}).get("failed_files") or 0)
    issues = {
        "duplicate_block_ids": duplicate_block_ids,
        "documents_without_blocks": sorted(documents_without_blocks),
        "empty_content_blocks": sorted(empty_content_blocks),
        "parser_failed_files": parser_failures,
    }
    return {
        "schema_version": 1,
        "passed": not any(
            (duplicate_block_ids, documents_without_blocks, empty_content_blocks, parser_failures)
        ),
        "counts": {
            "documents": len(documents),
            "pages": page_count,
            "blocks": block_count,
            "table_blocks": block_types.get("table", 0),
            "ocr_blocks": ocr_blocks,
            "block_types": dict(sorted(block_types.items())),
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit phase-0 parser output")
    parser.add_argument("--kb-path", type=Path, default=Path("output/knowledge_base.json"))
    parser.add_argument("--parser-report", type=Path, default=Path("output/parser_report.json"))
    parser.add_argument("--output", type=Path, default=Path("output/baseline/parser_audit.json"))
    args = parser.parse_args()

    audit = audit_knowledge_base(load_json(args.kb_path), load_json(args.parser_report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行测试并验证通过**

```powershell
python -m unittest tests.test_audit_parser_baseline -v
```

Expected: 4 tests run, 0 failures, 0 errors。

- [ ] **Step 5: 对当前解析产物执行审计**

```powershell
python scripts/audit_parser_baseline.py `
  --kb-path output/knowledge_base.json `
  --parser-report output/parser_report.json `
  --output output/baseline/parser_audit.json
$ParserAuditExit = $LASTEXITCODE
Get-Content -LiteralPath 'output/baseline/parser_audit.json'
exit $ParserAuditExit
```

Expected: 报告始终写出；无结构问题时退出 0，有重复 ID、空块、空文档或解析失败时退出 2。阶段 0 允许报告不通过，但不允许隐藏失败。

- [ ] **Step 6: 提交 Task 3**

```powershell
git add scripts/audit_parser_baseline.py tests/test_audit_parser_baseline.py
git commit -m "test: audit phase zero parser baseline"
```

---

### Task 4: 固定 Agentic RAG 分支并关闭模拟网络搜索

**Files:**

- Modify: `graph/graph.py:1-100`
- Modify: `graph/nodes/web_search.py:1-54`
- Create: `graph/tests/__init__.py`
- Create: `graph/tests/test_graph_routes.py`

**Interfaces:**

- Produces: `build_app()` 返回编译后的 LangGraph 应用
- Produces: `ExternalSearchDisabled` 明确表示外网搜索未配置
- Preserves: module-level `app = build_app()`，现有 `server.py` 和 `main.py` 不需要改调用方式

- [ ] **Step 0: 隔离已失效的 Agent 虚拟环境并创建可复现环境**

先只读取 `.venv/pyvenv.cfg`。只有其中记录的 Python 路径确实不存在时，才将旧环境改名保留，然后让 Poetry 使用本机 Python 3.11：

```powershell
$AgentRoot = 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
Set-Location $AgentRoot

$OldVenv = Join-Path $AgentRoot '.venv'
$BackupVenv = Join-Path $AgentRoot '.venv.broken-20260919'
if (Test-Path -LiteralPath "$OldVenv/pyvenv.cfg") {
    $Cfg = Get-Content -LiteralPath "$OldVenv/pyvenv.cfg"
    $ExecutableLine = $Cfg | Where-Object { $_ -like 'executable = *' } | Select-Object -First 1
    $RecordedPython = if ($ExecutableLine) { ($ExecutableLine -split '=', 2)[1].Trim() } else { $null }
    if ($RecordedPython -and -not (Test-Path -LiteralPath $RecordedPython)) {
        if (Test-Path -LiteralPath $BackupVenv) { throw "backup already exists: $BackupVenv" }
        Move-Item -LiteralPath $OldVenv -Destination $BackupVenv
    }
}

$Python311 = py -3.11 -c "import sys; print(sys.executable)"
if (-not (Test-Path -LiteralPath $Python311)) { throw 'Python 3.11 not found' }
poetry env use $Python311
poetry install --no-root
poetry run python --version
```

Expected: 最后一条输出 Python 3.11.x。备份目录不得删除；确认新环境和测试可运行后，后续由用户决定是否清理。

- [ ] **Step 1: 编写分支和失败关闭测试**

Create empty `graph/tests/__init__.py` and create `graph/tests/test_graph_routes.py`:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import graph.graph as graph_module
from graph.nodes.web_search import ExternalSearchDisabled, web_search


class Chain:
    def __init__(self, values):
        self.values = iter(values)

    def invoke(self, _payload):
        return next(self.values)


class AgentGraphRouteTests(unittest.TestCase):
    def _build(self, *, router_values, grounded_values, useful_values, counter):
        def retrieve(state):
            return {"question": state["question"], "documents": ["evidence"]}

        def grade(state):
            return {**state, "web_search": False}

        def generate(state):
            counter["generate"] += 1
            return {**state, "generation": f"draft-{counter['generate']}"}

        router = Chain([SimpleNamespace(datasource=value) for value in router_values])
        grounded = Chain([SimpleNamespace(binary_score=value) for value in grounded_values])
        useful = Chain([SimpleNamespace(binary_score=value) for value in useful_values])

        patches = patch.multiple(
            graph_module,
            retrieve=retrieve,
            grade_documents=grade,
            generate=generate,
            question_router=router,
            hallucination_grader=grounded,
            answer_grader=useful,
        )
        return patches

    def test_vectorstore_useful_path_generates_once(self):
        counter = {"generate": 0}
        with self._build(
            router_values=["vectorstore"],
            grounded_values=[True],
            useful_values=[True],
            counter=counter,
        ):
            result = graph_module.build_app().invoke({"question": "agent memory"})
        self.assertEqual(counter["generate"], 1)
        self.assertEqual(result["generation"], "draft-1")

    def test_unsupported_draft_retries_before_ending(self):
        counter = {"generate": 0}
        with self._build(
            router_values=["vectorstore"],
            grounded_values=[False, True],
            useful_values=[True],
            counter=counter,
        ):
            result = graph_module.build_app().invoke(
                {"question": "agent memory"},
                {"recursion_limit": 8},
            )
        self.assertEqual(counter["generate"], 2)
        self.assertEqual(result["generation"], "draft-2")

    def test_external_search_fails_closed(self):
        with self.assertRaisesRegex(ExternalSearchDisabled, "disabled"):
            web_search({"question": "today's market", "documents": []})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认 `build_app` 和异常类型不存在**

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
poetry run python -m unittest graph.tests.test_graph_routes -v
```

Expected: ImportError 或 AttributeError 指出 `build_app` / `ExternalSearchDisabled` 不存在。

- [ ] **Step 3: 将图构建收敛为 `build_app()` 并删除无条件结束边**

In `graph/graph.py`, keep the existing decision functions and replace the module-level graph construction with:

```python
def build_app():
    workflow = StateGraph(GraphState)
    workflow.add_node(RETRIEVE, retrieve)
    workflow.add_node(GRADE_DOCUMENTS, grade_documents)
    workflow.add_node(GENERATE, generate)
    workflow.add_node(WEBSEARCH, web_search)

    workflow.set_conditional_entry_point(
        route_question,
        {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
    )
    workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
    workflow.add_conditional_edges(
        GRADE_DOCUMENTS,
        decide_to_generate,
        {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
    )
    workflow.add_conditional_edges(
        GENERATE,
        grade_generation_grounded_in_documents_and_question,
        {"not supported": GENERATE, "useful": END, "not useful": WEBSEARCH},
    )
    workflow.add_edge(WEBSEARCH, GENERATE)
    return workflow.compile()


app = build_app()
```

Do not retain `workflow.add_edge(GENERATE, END)` because the conditional edge already owns every valid termination path.

- [ ] **Step 4: 用明确异常替换伪搜索实现**

Replace all content in `graph/nodes/web_search.py` with:

```python
from typing import Any, Dict

from graph.state import GraphState


class ExternalSearchDisabled(RuntimeError):
    """Raised when a graph requests external search without a real search provider."""


def web_search(state: GraphState) -> Dict[str, Any]:
    question = state.get("question", "")
    raise ExternalSearchDisabled(
        f"External web search is disabled; no verified provider is configured for: {question}"
    )
```

This file must not import `ChatOpenAI`, construct a search Prompt, catch the exception, or return an error string as a `Document`.

- [ ] **Step 5: 运行离线图测试并验证通过**

```powershell
poetry run python -m unittest graph.tests.test_graph_routes -v
```

Expected: 3 tests run, 0 failures, 0 errors；测试过程中不发起网络请求。

- [ ] **Step 6: 运行原链测试并记录它们是在线集成测试**

```powershell
poetry run pytest graph/chains/tests/test_chains.py -q
```

Expected: 有有效 API Key 和已构建 Chroma 时通过；缺少外部条件时将输出记录为 `blocked_external_dependency`，不能把它计为代码失败或成功。阶段 0 的硬门禁是 Step 5 的离线测试。

- [ ] **Step 7: 提交 Task 4**

Agent 参考目录当前不在主仓库 Git 根目录内。不要复制或伪造提交记录；保存 `git diff --no-index` 或文件哈希到阶段 0 manifest，并在后续阶段 1 迁入主仓库时提交。若用户先将该目录纳入独立 Git 仓库，再执行：

```powershell
git add graph/graph.py graph/nodes/web_search.py graph/tests
git commit -m "fix: fail closed when external search is unavailable"
```

---

### Task 5: 固化运行手册并执行阶段 0 验收

**Files:**

- Create: `docs/baseline/phase-0-runbook.md`
- Modify: `README.md`

**Interfaces:**

- Consumes: Tasks 1～4 的脚本和测试
- Produces: 可复制运行的 smoke/full 命令，以及 `output/baseline/` 下的阶段 0 产物

- [ ] **Step 1: 编写运行手册**

Create `docs/baseline/phase-0-runbook.md` with these sections and commands:

````markdown
# 阶段 0 基线运行手册

## 路径

```powershell
$FinanceRoot = 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
$AgentRoot = 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
```

## 离线硬门禁

```powershell
Set-Location $FinanceRoot
python -m unittest discover -s tests -v
python scripts/capture_phase0_manifest.py --agent-root $AgentRoot
python scripts/audit_parser_baseline.py

Set-Location $AgentRoot
poetry run python -m unittest graph.tests.test_graph_routes -v
```

`audit_parser_baseline.py` 退出 2 表示当前解析基线发现真实缺陷，报告仍然有效；必须保存问题，不能改成退出 0。

## 三案例 smoke

```powershell
Set-Location $FinanceRoot
python scripts/benchmark_regression.py --kb-path output/knowledge_base.json --test-set test/merged.xlsx --max-samples 3
```

## 全量基线

```powershell
Set-Location $FinanceRoot
$BaselineParse = Join-Path $FinanceRoot 'output/baseline/parsed'
$BaselineTemp = Join-Path $FinanceRoot 'output/baseline/temp'

python scripts/preprocess.py `
  --input data/documents `
  --output $BaselineParse `
  --tesseract-cmd "" `
  --temp-dir $BaselineTemp

python scripts/audit_parser_baseline.py `
  --kb-path "$BaselineParse/knowledge_base.json" `
  --parser-report "$BaselineParse/parser_report.json" `
  --output output/baseline/parser_audit_full.json

python scripts/benchmark_regression.py `
  --kb-path "$BaselineParse/knowledge_base.json" `
  --test-set test/merged.xlsx `
  --output output/baseline/qa_report_full.json `
  --final-output output/baseline/final_report_full.json `
  --max-samples 0
```

## 产物

- `output/baseline/manifest.json`
- `output/baseline/parser_audit.json`
- `output/baseline/qa_report.json`
- `output/baseline/final_report.json`
- 全量运行对应的 `*_full.json`

这些文件包含本地数据和答案，不进入 Git。对外展示前需要脱敏。

## 判读规则

- `evidence_response_rate` 只说明系统返回了答案和证据，不等于正确率。
- 解析审计失败是后续阶段的输入，不应修改报告来伪造通过。
- 在线链测试缺少 API Key 时标记为外部依赖阻塞；离线测试必须通过。
````

- [ ] **Step 2: 在 README 增加唯一入口**

Add immediately after the project introduction:

```markdown
## 融合项目实施基线

- 架构设计：[`docs/superpowers/specs/2026-09-19-financial-agentic-rag-integration-design.md`](docs/superpowers/specs/2026-09-19-financial-agentic-rag-integration-design.md)
- 阶段 0 运行手册：[`docs/baseline/phase-0-runbook.md`](docs/baseline/phase-0-runbook.md)

阶段 0 只冻结现状、记录基线并关闭模拟搜索；模型、向量库和 Agent 正式整合从后续阶段开始。
```

- [ ] **Step 3: 运行全部离线测试**

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
python -m unittest discover -s tests -v

Set-Location 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
poetry run python -m unittest graph.tests.test_graph_routes -v
```

Expected: 金融项目 10 tests、Agent 图 3 tests，合计 13 tests，0 failures，0 errors。

- [ ] **Step 4: 运行 smoke 基线并检查产物结构**

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
$AgentRoot = 'C:\code\个人资料\code\Agentic_RAG\Agentic_RAG\0_agent项目_Agentic RAG v3\代码\agentic-rag'
python scripts/capture_phase0_manifest.py --agent-root $AgentRoot
python scripts/audit_parser_baseline.py
python scripts/benchmark_regression.py --max-samples 3

$Manifest = Get-Content output/baseline/manifest.json -Raw | ConvertFrom-Json
$Audit = Get-Content output/baseline/parser_audit.json -Raw | ConvertFrom-Json
$Qa = Get-Content output/baseline/qa_report.json -Raw | ConvertFrom-Json

if ($Manifest.schema_version -ne 1) { throw 'manifest schema mismatch' }
if ($Audit.schema_version -ne 1) { throw 'parser audit schema mismatch' }
if ($Qa.schema_version -ne 1) { throw 'QA report schema mismatch' }
if ($Qa.cases.Count -ne 3) { throw 'smoke report must contain exactly 3 cases' }
```

Expected: 结构断言通过。解析审计若返回 2，应单独记录为已发现基线问题后继续查看报告；不得忽略或重写其退出码。

- [ ] **Step 5: 检查没有提交秘密和大体积产物**

```powershell
Set-Location 'C:\code\个人资料\code\Agentic_RAG\finance-qa-platform'
git status --short
git diff --check
git diff --cached --check
git ls-files .env output vector_cache data_app.db
```

Expected: 最后一条命令不列出 `.env`、`output/` 产物、`vector_cache/` 或 `data_app.db`；diff check 无错误。

- [ ] **Step 6: 提交文档和 README**

```powershell
git add README.md docs/baseline/phase-0-runbook.md `
  docs/superpowers/specs/2026-09-19-financial-agentic-rag-integration-design.md `
  docs/superpowers/plans/2026-09-19-phase-0-baseline-plan.md
git commit -m "docs: define phase zero integration baseline"
```

- [ ] **Step 7: 阶段 0 完成检查**

执行以下验收，不满足任何一项都不能进入阶段 1：

1. 环境 manifest 存在，包含两个代码目录、依赖、知识库和测试集哈希，且没有秘密值。
2. 金融 QA 基线逐案例保留答案、证据、错误和延迟，指标名称不暗示正确性。
3. 解析审计明确列出空文档、空块、重复 block ID 和失败文件。
4. Agent 图离线分支测试通过，第一次无依据时确实会重试生成。
5. 外部搜索未配置时明确失败，不生成模拟搜索 Evidence。
6. 生成产物处于 `output/baseline/` 且未进入 Git。
7. 运行手册能从干净终端执行 smoke 流程。
8. 阶段 0 没有引入新向量库、后台队列、文档解析器或模型。

---

## 阶段 0 结束后的下一步

阶段 0 通过后，依据实际基线报告为“阶段 1：统一项目与核心数据契约”单独编写实施计划。阶段 1 只处理 DocIR、Evidence、Run、ModelSnapshot、ModelGateway 和兼容 API，不提前实施阶段 2～7。
