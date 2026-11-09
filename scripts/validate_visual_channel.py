"""Offline candidate visual coverage and real retrieval validation."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.system.visual_pages import build_visual_pages
from src.pipeline.snapshots import read_manifest, sha256_file


def verify_http(manifest_path, hits):
    """Real evidence API, disposable database; no active snapshot registration."""
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.api.evidence import router
    from app.core.deps import get_current_user
    from app.db.session import get_db
    from app.db.base import Base
    from app.models import DocumentVersionRecord, EvidenceRecord, KnowledgeSnapshotRecord, User
    from src.contracts import evidence_from_v2
    from app.services.run_service import RunService
    import uuid
    manifest = read_manifest(manifest_path)
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    try:
        with sessions() as db:
            db.add(User(id=1, username='visual-audit', email='audit@example.com', password_hash='unused', role='user'))
            db.add(KnowledgeSnapshotRecord(id=manifest.snapshot_id, manifest_path=str(manifest_path.resolve()),
                manifest_sha256=manifest.manifest_sha256, document_versions_json=json.dumps([a.document_version for a in manifest.versions])))
            for a in manifest.versions:
                db.add(DocumentVersionRecord(id=a.document_version, document_id=a.document_id,
                    source_sha256='audit', source_size_bytes=0, source_format='audit', parser_name='audit',
                    parser_version='audit', parser_config_hash='audit', chunker_version='audit', status='active',
                    docir_path=str((manifest_path.parent/a.docir_path).resolve()), quality_report_path=''))
            db.commit()
        app = FastAPI()
        app.include_router(router, prefix='/api')
        def database():
            with sessions() as db:
                yield db
        app.dependency_overrides[get_db] = database
        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id=1, role='user')
        checked = []
        with TestClient(app) as client:
            for hit in hits:
                evidence = RunService._scope_evidence(evidence_from_v2(hit, manifest.snapshot_id),
                    run_id=str(uuid.uuid4()), user_id=1)
                with sessions() as db:
                    RunService._insert_evidence(db, evidence, run_id=None, user_id=1)
                    db.commit()
                detail = client.get(f'/api/evidence/{evidence.evidence_id}')
                response = client.get(f'/api/evidence/{evidence.evidence_id}/image')
                if detail.status_code != 200 or response.status_code != 200:
                    raise AssertionError(f'HTTP evidence check failed: {detail.status_code}/{response.status_code}')
                if hashlib.sha256(response.content).hexdigest() != hit['image_sha256']:
                    raise AssertionError('HTTP image bytes mismatch')
                checked.append({'evidence_id': evidence.evidence_id, 'page_id': hit['page_id'],
                                'image_sha256': hit['image_sha256'], 'status': 'passed'})
        return checked
    finally:
        engine.dispose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kb-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--model', default=str(ROOT.parent/'model_cache/chinese-clip-vit-base-patch16'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args(argv)
    for key in ('FINANCE_DISABLE_PAID_API', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        os.environ[key] = '1'
    root = args.kb_dir.resolve()
    if root == (ROOT/'output/kb_v3').resolve():
        parser.error('candidate directory required; active KB is protected')
    manifest_path = root/'kb_v3_manifest.json'
    chunks_path = root/'chunks.jsonl'
    before = {'manifest': sha256_file(manifest_path), 'chunks': sha256_file(chunks_path)}
    blocks = [json.loads(line) for line in chunks_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    pages = build_visual_pages(manifest_path, blocks, allowed_roots=(root, ROOT/'output/kb_v3/parsed'))
    inventory = [{**p, 'chunks': [b.get('chunk_id') or b['block_id'] for b in p['chunks']]} for p in pages]
    (root/'visual_pages.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(snapshot_id=read_manifest(manifest_path).snapshot_id, physical_pages=len(pages),
        text_chunks=len(blocks), unlinked_pages=sum(not p['chunks'] for p in pages),
        missing_images=sum(p['image_sha256'] is None for p in pages), visual_ready=False,
        audit_only=args.audit_only, source_hashes=before, page_results=[], queries=[])
    if not args.audit_only:
        import numpy as np
        from src.system.vision import ChineseVisionRetriever
        r = ChineseVisionRetriever(args.cache_dir or root/'vector_cache', device=args.device, model_name=args.model)
        r.build(blocks, {}, visual_pages=pages)
        encoded = {p['page_id']: i for i, p in enumerate(r.pages)}
        finite = bool(np.isfinite(r.vectors).all())
        normalized = bool(np.allclose(np.linalg.norm(r.vectors, axis=1), 1, atol=1e-5))
        report.update(stats=r.stats, vectors_finite=finite, vectors_normalized=normalized,
            visual_ready=len(encoded) == len(pages) and finite and normalized)
        report['page_results'] = [{'page_id': p['page_id'], 'document_version': p['document_version'],
            'page': p['page'], 'image_sha256': p['image_sha256'], 'unlinked': not p['chunks'],
            'vector_row': encoded.get(p['page_id']),
            'status': 'encoded' if p['page_id'] in encoded else 'failed'} for p in pages]
        natural_hits = []
        for query in ['农业银行2024年一季度营业收入和净利润', '银行资产负债表', '手机银行转账操作步骤',
                      '银行业金融机构资产规模增长', '理财产品走势图']:
            hits = r.search(query, top_k=5)
            report['queries'].append({'query': query, 'hits': hits})
            natural_hits.extend(hits)
        report['natural_query_http'] = verify_http(manifest_path, natural_hits)
        targets = [p for p in r.pages if not p['chunks'] and
                   (p['source_doc'] == '银行业金融机构资产规模稳步增长.jpg' or
                    ('理财产品对比表' in p['source_doc'] and p['page'] == 2))]
        deterministic = []
        for page in targets:
            scores = np.full(len(r.pages), -1.0)
            scores[encoded[page['page_id']]] = 1.0
            deterministic.extend(r.resolve_hits(scores, top_k=1))
        report['deterministic_identity_http_not_relevance'] = verify_http(manifest_path, deterministic)
    else:
        report['page_results'] = [{'page_id': p['page_id'], 'status': 'missing_image' if p['image_sha256'] is None
                                  else 'inventoried_not_encoded'} for p in pages]
    if before != {'manifest': sha256_file(manifest_path), 'chunks': sha256_file(chunks_path)}:
        raise AssertionError('source artifacts changed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('physical_pages', 'unlinked_pages', 'missing_images', 'visual_ready')}, ensure_ascii=False))
    return 2 if report['missing_images'] or (not args.audit_only and not report['visual_ready']) else 0


if __name__ == "__main__":
    raise SystemExit(main())
