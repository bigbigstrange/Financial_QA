"""Validate and register an on-disk KB snapshot; activate only with --activate.

Existing snapshots/artifacts are retained. SQLite is backed up before mutation.
"""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.db.session import SessionLocal, engine
from app.models import DocumentRecord, DocumentVersionRecord, KnowledgeSnapshotRecord
from src.contracts import utc_now
from src.pipeline.docir import DocIR
from src.pipeline.snapshots import read_manifest, sha256_file


def same_evidence_content(left, right):
    """Relocated image/source paths do not require rewriting historical versions."""
    def signature(doc):
        return (doc.document_id, doc.document_version, doc.source_hash,
                [(b.block_id, b.block_type, b.text, b.page, b.bbox,
                  b.section_path, b.structural_locator, b.table_md, b.table_rows,
                  b.table_headers, b.unit) for b in doc.all_blocks()])
    return signature(left) == signature(right)


def validate(path):
    manifest = read_manifest(path)
    documents = []
    for artifact in manifest.versions:
        docpath = Path(artifact.docir_path.replace('\\', '/'))
        if not docpath.is_absolute():
            docpath = path.parent / docpath
        docpath = docpath.resolve()
        if sha256_file(docpath) != artifact.docir_sha256:
            raise ValueError(f"DocIR hash mismatch: {docpath}")
        doc = DocIR.from_dict(json.loads(docpath.read_text(encoding="utf-8")))
        if (doc.document_id, doc.document_version) != (artifact.document_id, artifact.document_version):
            raise ValueError(f"DocIR identity mismatch: {docpath}")
        documents.append((doc, docpath))
    if not documents:
        raise ValueError("Cannot register an empty snapshot")
    return manifest, documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest, documents = validate(path)
    if engine.dialect.name == "sqlite" and engine.url.database:
        backup = ROOT / "output/backups" / f"catalog-{time.time_ns()}.sqlite3"
        backup.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(engine.url.database) as source, sqlite3.connect(backup) as target:
            source.backup(target)
        print(f"Backup: {backup}", flush=True)
    with SessionLocal.begin() as db:
        for doc, docpath in documents:
            record = db.get(DocumentRecord, doc.document_id)
            if record is None:
                record = DocumentRecord(id=doc.document_id, display_name=doc.source_name or doc.doc_id,
                                        status="active", access_scope_json=json.dumps(doc.access_scope))
                db.add(record)
                db.flush()
            if record.status == "deleted":
                raise ValueError(f"Refusing to reactivate deleted document: {doc.document_id}")
            version = db.get(DocumentVersionRecord, doc.document_version)
            if version is None:
                version = DocumentVersionRecord(id=doc.document_version, document_id=doc.document_id,
                    source_sha256=doc.source_hash, source_size_bytes=doc.source_size_bytes,
                    source_format=doc.source_format, parser_name=doc.parser_name,
                    parser_version=doc.parser_version, parser_config_hash=doc.parser_config_hash,
                    chunker_version=doc.chunker_version, docir_path=str(docpath), status="ready")
                db.add(version)
            elif Path(version.docir_path or "").resolve() != docpath:
                original_path = Path(version.docir_path or "")
                if not original_path.is_file():
                    raise ValueError(f"Historical artifact missing: {doc.document_version}")
                original = DocIR.from_dict(json.loads(original_path.read_text(encoding="utf-8")))
                if not same_evidence_content(original, doc):
                    raise ValueError(f"Version reused with changed evidence: {doc.document_version}")
                # Preserve the historical canonical artifact and references.
            if args.activate:
                if record.active_version_id and record.active_version_id != version.id:
                    previous = db.get(DocumentVersionRecord, record.active_version_id)
                    if previous:
                        previous.status = "superseded"
                record.active_version_id = version.id
                version.status = "active"
                version.activated_at = utc_now()
        if args.activate:
            for old in db.query(KnowledgeSnapshotRecord).filter_by(status="active").all():
                old.status = "retired"
        snapshot = db.get(KnowledgeSnapshotRecord, manifest.snapshot_id)
        if snapshot is None:
            snapshot = KnowledgeSnapshotRecord(id=manifest.snapshot_id)
            db.add(snapshot)
        snapshot.manifest_path = str(path)
        snapshot.status = "active" if args.activate else snapshot.status or "candidate"
        snapshot.document_versions_json = json.dumps([v.document_version for v in manifest.versions])
        snapshot.parser_schema_version = manifest.config.docir_schema_version
        snapshot.chunker_version = manifest.config.chunker_version
        snapshot.block_count = manifest.block_count
        snapshot.manifest_sha256 = manifest.manifest_sha256
        if args.activate:
            snapshot.activated_at = utc_now()
    print(json.dumps({"snapshot": manifest.snapshot_id, "documents": len(documents), "activated": args.activate}))


if __name__ == "__main__":
    main()
