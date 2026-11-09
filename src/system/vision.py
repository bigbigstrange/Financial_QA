"""Chinese page-image retrieval with real visual embeddings and chunk provenance.

Missing/unreadable images are reported and skipped; no text vectors are mixed
into the image space. Page hits resolve to existing chunks for evidence APIs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def page_image_key(block, page, page_images):
    """Prefer immutable version identity; accept older document/name-keyed maps."""
    identities = [block.get("document_version"), block.get("document_id"), block["source_doc"]]
    keys = [(identity, page) for identity in identities if identity and identity != "legacy-v1"]
    return next((key for key in keys if key in page_images), keys[0])


class ChineseVisionRetriever:
    def __init__(self, cache_dir, *, device="cpu", model_name=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.model_name = model_name or os.environ.get(
            "FINANCE_VISION_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
        if Path(self.model_name).is_dir():
            self.model_name = str(Path(self.model_name).resolve())
        self.model = self.processor = None
        self.pages = []
        self.vectors = None
        self.stats = {"backend": "chinese_clip", "indexed_pages": 0, "skipped_pages": []}

    def _load_model(self):
        if self.model is None:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            self.processor = ChineseCLIPProcessor.from_pretrained(self.model_name)
            self.model = ChineseCLIPModel.from_pretrained(self.model_name).to(self.device).eval()

    @staticmethod
    def page_items(blocks, page_images):
        grouped = {}
        for block in blocks:
            start = block.get("page_start") or block.get("page")
            if start is None:
                continue
            end = block.get("page_end") or start
            for page in range(int(start), int(end) + 1):
                image_key = page_image_key(block, page, page_images)
                if image_key not in page_images:
                    continue
                version = block.get("document_version")
                identity = (version if version and version != "legacy-v1" else
                            block.get("document_id") or block["source_doc"])
                bucket = grouped.setdefault((identity, page), {
                    "image_ref": str(page_images[image_key]), "chunks": []})
                bucket["chunks"].append(block)
        return [{"source_doc": item["chunks"][0]["source_doc"],
                 "document_id": item["chunks"][0].get("document_id"),
                 "document_version": item["chunks"][0].get("document_version"),
                 "page": page, "image_ref": item["image_ref"],
                 "chunks": item["chunks"]}
                for (_, page), item in sorted(grouped.items())]

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def build(self, blocks, page_images, force_rebuild=False, *, visual_pages=None):
        from src.pipeline.snapshots import sha256_file
        pages = self.page_items(blocks, page_images) if visual_pages is None else visual_pages
        model_dir = Path(self.model_name)
        model_files = {p.name: sha256_file(p) for p in sorted(model_dir.glob('*'))
                       if p.is_file() and p.suffix in {'.json', '.bin', '.safetensors', '.txt'}}
        fingerprint = {"schema": 2, "model": self.model_name, "model_files": model_files,
                       "preprocessing": "ChineseCLIPProcessor:RGB:text512:l2", "pages": []}
        self.stats['skipped_pages'] = []
        for page in pages:
            path = Path(page["image_ref"]) if page['image_ref'] else Path('__missing_image__')
            stat = path.stat() if path.is_file() else None
            fingerprint["pages"].append([page["source_doc"], page["page"], str(path),
                stat.st_size if stat else 0, stat.st_mtime_ns if stat else 0,
                [b.get("chunk_id") or b["block_id"] for b in page["chunks"]],
                page.get('page_id'), page.get('image_sha256')])
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:32]
        cache = self.cache_dir / f"vision_cn_{key}.npz"
        self.stats['fingerprint'] = key
        if cache.exists() and not force_rebuild:
            with np.load(cache, allow_pickle=False) as saved:
                self.vectors = saved["vectors"]
                indices = saved["indices"].tolist()
                self.pages = [pages[i] for i in indices]
            self.stats["indexed_pages"] = len(self.pages)
            self.stats["skipped_pages"] = [p["image_ref"] for i, p in enumerate(pages) if i not in set(indices)]
            return
        self._load_model()
        import torch
        from PIL import Image
        vectors, indices = [], []
        batch_size = max(1, int(os.environ.get("FINANCE_VISION_BATCH_SIZE", "4")))
        for start in range(0, len(pages), batch_size):
            images, valid = [], []
            for i in range(start, min(start + batch_size, len(pages))):
                try:
                    if not pages[i]['image_ref']:
                        raise ValueError('missing_page_image')
                    with Image.open(pages[i]["image_ref"]) as image:
                        images.append(image.convert("RGB"))
                    valid.append(i)
                except (OSError, ValueError) as exc:
                    self.stats["skipped_pages"].append({"path": pages[i]["image_ref"], "error": str(exc)})
            if not images:
                continue
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.inference_mode():
                    encoded = self.model.get_image_features(**inputs).cpu().numpy()
                vectors.extend(encoded)
                indices.extend(valid)
            finally:
                for image in images:
                    image.close()
            if start % (batch_size * 50) == 0:
                print(f"[vision] encoded {len(indices)}/{len(pages)} pages", flush=True)
        if not vectors:
            raise RuntimeError("No usable visual embeddings; image channel was not enabled")
        self.vectors = self._normalize(vectors)
        self.pages = [pages[i] for i in indices]
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, vectors=self.vectors, indices=np.asarray(indices))
        tmp.replace(cache)
        self.stats["indexed_pages"] = len(self.pages)

    def search(self, query, top_k=6, visibility=None):
        if self.vectors is None or top_k <= 0:
            return []
        self._load_model()
        import torch
        inputs = self.processor(text=[query], padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            q = self._normalize(self.model.get_text_features(**inputs).cpu().numpy())[0]
        scores = self.vectors @ q
        return self.resolve_hits(scores, top_k=top_k, visibility=visibility, query=query)

    def resolve_hits(self, scores, top_k=6, visibility=None, query=""):
        if top_k <= 0:
            return []
        out, seen, seen_pages = [], set(), set()
        query_terms = {
            query[i:i + 2]
            for i in range(len(query) - 1)
            if all("\u4e00" <= char <= "\u9fff" for char in query[i:i + 2])
        }
        for index in np.argsort(-scores):
            page = self.pages[int(index)]
            version = page.get('document_version')
            page_key = (
                (version if version and version != 'legacy-v1' else
                 page.get('document_id') or page.get('source_doc')),
                page.get('page'),
            )
            if page_key in seen_pages:
                continue
            candidates = page['chunks'] or [{
                **{k: page[k] for k in ('snapshot_id', 'document_id',
                    'document_version', 'source_doc', 'quality_flags')},
                'block_id': page['page_id'], 'evidence_kind': 'page_image',
                'chunk_id': None, 'block_ids': [], 'content': '', 'page': page['page']}]
            eligible = [
                block for block in candidates
                if (block.get("chunk_id") or block["block_id"]) not in seen
                and (visibility is None or visibility.allowed(block.get("document_version") or ""))
            ]
            if not eligible:
                continue

            def overlap(block):
                identity = ' '.join(str(block.get(field) or '') for field in
                                    ('content', 'section_path', 'source_doc'))
                return sum(term in identity for term in query_terms)

            block = max(eligible, key=overlap)
            key = block.get("chunk_id") or block["block_id"]
            seen.add(key)
            seen_pages.add(page_key)
            out.append({**block, "block_id": key,
                        "chunk_id": None if block.get('evidence_kind') == 'page_image' else key,
                        "page_id": page.get('page_id'), "image_sha256": page.get('image_sha256'),
                        "score": float(scores[index]), "image_score": float(scores[index]),
                        "image_ref": page["image_ref"], "vision_page": page["page"],
                        "retrieval_channel": "image_chinese_clip",
                        "retrieval_channels": ["image_chinese_clip"]})
            if len(out) >= top_k:
                return out
        return out
