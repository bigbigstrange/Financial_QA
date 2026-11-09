"""Verified physical-page inventory, independent of text chunk availability."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.pipeline.docir import DocIR
from src.pipeline.snapshots import read_manifest, sha256_file


def make_visual_page_id(snapshot_id: str, document_id: str,
                        document_version: str, page_no: int) -> str:
    if not all(isinstance(x, str) and x.strip() for x in
               (snapshot_id, document_id, document_version)):
        raise ValueError('invalid_visual_identity')
    if type(page_no) is not int or page_no < 1:
        raise ValueError('invalid_physical_page')
    identity = json.dumps([snapshot_id, document_id, document_version, page_no],
                          ensure_ascii=False, separators=(',', ':'))
    return 'vp_' + hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]


def resolve_artifact_path(reference: str, base_dir: Path,
                          allowed_roots: tuple[Path, ...]) -> Path:
    if not reference or not isinstance(reference, str):
        raise ValueError('invalid_artifact_reference')
    # Keep immutable manifest bytes/hash; normalize Windows separators only at lookup.
    resolved = (base_dir / reference.replace('\\', '/')).resolve()
    if not any(resolved.is_relative_to(root.resolve()) for root in allowed_roots):
        raise ValueError('image_path_outside_artifact_roots')
    return resolved


def build_visual_pages(manifest_path: Path, blocks: list[dict], *,
                       allowed_roots: tuple[Path, ...],
                       selected_versions: set[str] | None = None,
                       source_keywords: list[str] | None = None) -> list[dict]:
    manifest_path = Path(manifest_path)
    manifest = read_manifest(manifest_path)
    artifacts = {item.document_version: item for item in manifest.versions}
    if len(artifacts) != len(manifest.versions):
        raise ValueError('duplicate_document_version')
    linked = defaultdict(list)
    for block in blocks:
        version = block.get('document_version')
        artifact = artifacts.get(version)
        if (artifact is None or block.get('document_id') != artifact.document_id
                or block.get('snapshot_id') != manifest.snapshot_id):
            raise ValueError('chunk_identity_mismatch')
        start = block.get('page_start') or block.get('page')
        end = block.get('page_end') or start
        if start is None:
            continue
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            raise ValueError('invalid_chunk_page_range')
        for page in range(start, end + 1):
            linked[(version, page)].append(block)

    pages = []
    keywords = [keyword.casefold() for keyword in (source_keywords or []) if keyword]
    for artifact in manifest.versions:
        if not artifact.document_id or not artifact.document_version:
            raise ValueError('invalid_document_identity')
        selected = selected_versions is None or artifact.document_version in selected_versions
        if not selected and not keywords:
            continue
        path = resolve_artifact_path(artifact.docir_path, manifest_path.parent, allowed_roots)
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact.docir_sha256:
            raise ValueError('docir_hash_mismatch')
        doc = DocIR.from_dict(json.loads(data))
        if (doc.document_id != artifact.document_id or
                doc.document_version != artifact.document_version):
            raise ValueError('docir_identity_mismatch')
        if not selected and not any(keyword in (doc.source_name or Path(doc.source_path).name).casefold()
                                    for keyword in keywords):
            continue
        seen = set()
        for page in doc.pages:
            if page.page_no is None:
                continue
            page_id = make_visual_page_id(manifest.snapshot_id, doc.document_id,
                                          doc.document_version, page.page_no)
            if page_id in seen:
                raise ValueError('duplicate_physical_page')
            seen.add(page_id)
            chunks = linked[(doc.document_version, page.page_no)]
            flags = []
            if not page.blocks:
                flags.append('no_parsed_blocks')
            if not chunks:
                flags.append('no_text_chunks')
            reference = page.meta.get('image_ref')
            image_path = (resolve_artifact_path(reference, manifest_path.parent, allowed_roots)
                          if reference else None)
            digest = None
            if image_path is None or not image_path.is_file():
                flags.append('missing_page_image')
            else:
                digest = sha256_file(image_path)
            pages.append(dict(page_id=page_id, snapshot_id=manifest.snapshot_id,
                document_id=doc.document_id, document_version=doc.document_version,
                source_doc=doc.source_name or Path(doc.source_path).name, page=page.page_no,
                image_ref=str(image_path) if image_path else None, image_sha256=digest,
                chunks=chunks, quality_flags=flags))
    return pages
