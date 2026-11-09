"""Read-only structural audit; never activates a candidate or claims answer accuracy."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline.docir import DocIR
from src.pipeline.snapshots import read_manifest, sha256_file


def audit(kb_dir):
    kb_dir = Path(kb_dir).resolve()
    manifest = read_manifest(kb_dir / "kb_v3_manifest.json")
    errors, missing_images = [], []
    versions, global_blocks, seen_chunks = {}, set(), set()
    physical_pages = image_pages = chunk_count = 0
    for artifact in manifest.versions:
        path = Path(artifact.docir_path)
        if not path.is_absolute():
            path = kb_dir / path
        if not path.is_file() or sha256_file(path) != artifact.docir_sha256:
            errors.append({"version": artifact.document_version, "reason": "artifact_missing_or_changed"})
            continue
        doc = DocIR.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if (doc.document_id, doc.document_version) != (artifact.document_id, artifact.document_version):
            errors.append({"version": artifact.document_version, "reason": "identity_mismatch"})
        blocks = doc.all_blocks()
        ids = [b.block_id for b in blocks]
        if len(ids) != len(set(ids)) or global_blocks.intersection(ids):
            errors.append({"version": artifact.document_version, "reason": "duplicate_block_id"})
        global_blocks.update(ids)
        versions[doc.document_version] = {"document_id": doc.document_id, "blocks": set(ids),
                                           "pages": {p.page_no for p in doc.pages if p.page_no is not None}}
        for page in doc.pages:
            if page.page_no is None:
                continue
            physical_pages += 1
            ref = (page.meta or {}).get("image_ref")
            resolved = Path(ref) if ref else None
            if resolved is not None and not resolved.is_absolute():
                resolved = kb_dir / resolved
            if resolved is not None and resolved.is_file():
                image_pages += 1
            else:
                missing_images.append({"version": doc.document_version, "page": page.page_no})
    with (kb_dir / "chunks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            chunk_count += 1
            cid = row.get("chunk_id")
            version = versions.get(row.get("document_version"))
            reasons = []
            if not cid or cid in seen_chunks:
                reasons.append("duplicate_or_missing_chunk_id")
            seen_chunks.add(cid)
            if row.get("snapshot_id") != manifest.snapshot_id:
                reasons.append("snapshot_mismatch")
            if not version or row.get("document_id") != version["document_id"]:
                reasons.append("document_identity_mismatch")
            elif not row.get("block_ids") or not set(row["block_ids"]).issubset(version["blocks"]):
                reasons.append("original_blocks_missing")
            if not (row.get("content") or "").strip():
                reasons.append("empty_content")
            if version:
                for key in ("page_start", "page_end"):
                    if row.get(key) is not None and row[key] not in version["pages"]:
                        reasons.append("page_not_in_document")
            if reasons:
                errors.append({"chunk_id": cid, "reasons": reasons})
    return {"snapshot_id": manifest.snapshot_id, "documents": len(manifest.versions),
            "chunks": chunk_count, "unique_blocks": len(global_blocks),
            "structural_integrity_passed": not errors, "errors": errors,
            "physical_pages": physical_pages, "available_page_images": image_pages,
            "missing_page_images": missing_images,
            "full_corpus_coverage_verified": False, "answer_accuracy_verified": False,
            "activated": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.kb_dir)
    output = args.kb_dir / "integrity_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**{k: v for k, v in report.items() if k not in {"errors", "missing_page_images"}},
                      "error_count": len(report["errors"]), "missing_image_count": len(report["missing_page_images"])}, ensure_ascii=False))
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
