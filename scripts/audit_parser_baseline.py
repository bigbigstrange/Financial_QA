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
    empty_knowledge_base = not documents
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
        "empty_knowledge_base": empty_knowledge_base,
        "duplicate_block_ids": duplicate_block_ids,
        "documents_without_blocks": sorted(documents_without_blocks),
        "empty_content_blocks": sorted(empty_content_blocks),
        "parser_failed_files": parser_failures,
    }
    return {
        "schema_version": 1,
        "passed": not any(
            (
                empty_knowledge_base,
                duplicate_block_ids,
                documents_without_blocks,
                empty_content_blocks,
                parser_failures,
            )
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
