import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from tests.unit import test_visual_pages
from src.pipeline.snapshots import sha256_file


class VisualValidationCLITests(unittest.TestCase):
    def test_http_validation_uses_real_route_for_page_evidence(self):
        from tests.integration import test_evidence_api
        from scripts.validate_visual_channel import verify_http
        fixture = test_evidence_api.EvidenceAPITests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture._add_image()
        from app.models import EvidenceRecord
        with fixture.Session() as db:
            record = db.get(EvidenceRecord, 'ev-word')
            saved = json.loads(record.evidence_json)
            hit = {**saved['metadata'], 'block_id': record.block_id, 'content': '',
                   'document_id': 'doc-word', 'document_version': 'dv-word', 'page': 2}
        result = verify_http(fixture.tmp_path/'manifest.json', [hit])
        self.assertEqual(result[0]['status'], 'passed')
        self.assertEqual(result[0]['image_sha256'], sha256_file(fixture.tmp_path/'2.png'))

    def test_audit_reports_all_pages_and_preserves_sources(self):
        fixture = test_visual_pages.VisualPagesTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        chunks = fixture.root/'chunks.jsonl'
        chunks.write_text('\n'.join(json.dumps(b) for b in fixture.blocks), encoding='utf-8')
        before = (sha256_file(chunks), sha256_file(fixture.path))
        report = fixture.root/'audit.json'
        command = [sys.executable, 'scripts/validate_visual_channel.py', '--kb-dir', str(fixture.root),
                   '--report', str(report), '--audit-only']
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
            env={**os.environ, 'FINANCE_DISABLE_PAID_API': '1', 'HF_HUB_OFFLINE': '1',
                 'TRANSFORMERS_OFFLINE': '1', 'PYTHONIOENCODING': 'utf-8'})
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(data['physical_pages'], 2)
        self.assertEqual(data['unlinked_pages'], 1)
        self.assertFalse(data['visual_ready'])
        self.assertEqual(before, (sha256_file(chunks), sha256_file(fixture.path)))
        (fixture.root/'b.png').unlink()
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2)
        data = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(data['missing_images'], 1)
        self.assertFalse(data['visual_ready'])
