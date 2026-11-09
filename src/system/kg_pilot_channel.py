"""Opt-in retrieval of locally verified, source-aligned KG pilot triples."""
from __future__ import annotations

import re
from typing import Any

from scripts.build_budgeted_kg import GENERIC_ENTITIES


class KGPilotRetriever:
    def __init__(self, blocks: list[dict], checkpoint: Any):
        self.blocks = {
            str(block.get("chunk_id") or block.get("block_id")): block
            for block in blocks
        }
        self.triples = checkpoint.completed_triples()

    def search(self, query: str, top_k: int = 10, visibility=None) -> list[dict]:
        normalized = re.sub(r"\s+", "", query).casefold()
        found: dict[str, dict] = {}
        for triple in self.triples:
            chunk_id = str(triple.get("chunk_id") or "")
            block = self.blocks.get(chunk_id)
            if not block or triple.get("snapshot_id") != block.get("snapshot_id"):
                continue
            if triple.get("document_version") != block.get("document_version"):
                continue
            if visibility is not None and not visibility.allowed(block.get("document_version") or ""):
                continue
            head = re.sub(r"\s+", "", triple.get("head", "")).casefold()
            tail = re.sub(r"\s+", "", triple.get("tail", "")).casefold()
            relation = re.sub(r"\s+", "", triple.get("relation", "")).casefold()
            relation_topic = relation.rstrip("为是")
            if (triple.get("head") in GENERIC_ENTITIES or
                    triple.get("tail") in GENERIC_ENTITIES or
                    triple.get("relation") in {"标题为", "为"}):
                continue
            if not head or not tail or head not in normalized:
                continue
            if not (tail in normalized or relation in normalized or
                    (len(relation_topic) >= 2 and relation_topic in normalized)):
                continue
            evidence = triple.get("evidence") or ""
            if (not evidence or evidence not in block.get("content", "") or
                    any(not triple.get(key) or triple[key] not in evidence
                        for key in ("head", "relation", "tail"))):
                continue
            item = dict(block)
            item["block_id"] = block.get("block_id") or chunk_id
            item["graph_score"] = 1.0
            item["score"] = 1.0
            item["retrieval_channel"] = "kg_pilot"
            item["kg_entities"] = [triple["head"], triple["tail"]]
            item["kg_relations"] = [triple["relation"]]
            item["kg_evidence"] = triple["evidence"]
            found[chunk_id] = item
            if len(found) >= top_k:
                break
        return list(found.values())
