import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class KGExtractor:
    """
    轻量离线知识抽取器（阶段2）：
    - 实体：关键词 + 数值实体
    - 关系：同块共现构边（co_occurs_with）
    - 属性：记录来源文档、页码、块号
    """

    ENTITY_KEYWORDS = [
        "中国人民银行",
        "银保监会",
        "商业银行",
        "贷款",
        "利率",
        "存款",
        "汇率",
        "人民币",
        "外汇",
        "房地产",
        "财政",
        "客户经理",
    ]

    NUMBER_PATTERN = re.compile(r"(?<!\d)(\d+(?:\.\d+)?%?)")

    def extract_from_kb(self, kb_data: Dict) -> Tuple[List[Dict], List[Dict], Dict]:
        entity_map: Dict[str, Dict] = {}
        triples: List[Dict] = []
        doc_entity_index: Dict[str, set] = defaultdict(set)
        page_entity_index: Dict[str, set] = defaultdict(set)

        for doc in kb_data.get("documents", []):
            doc_id = doc.get("doc_id", "unknown_doc")
            for page in doc.get("pages", []):
                page_no = int(page.get("page_no", 1))
                for block in page.get("blocks", []):
                    text = (block.get("text") or "").strip()
                    if not text:
                        continue
                    block_id = block.get("block_id", f"{doc_id}_p{page_no}_b")
                    entities = self._extract_entities(text, doc_id, page_no, block_id)
                    entity_ids = []
                    for ent in entities:
                        eid = ent["entity_id"]
                        entity_ids.append(eid)
                        if eid not in entity_map:
                            entity_map[eid] = ent
                        else:
                            entity_map[eid]["mentions"].append(ent["mentions"][0])
                        doc_entity_index[doc_id].add(eid)
                        page_entity_index[f"{doc_id}::{page_no}"].add(eid)

                    # 共现关系
                    for i in range(len(entity_ids)):
                        for j in range(i + 1, len(entity_ids)):
                            h = entity_ids[i]
                            t = entity_ids[j]
                            triples.append(
                                {
                                    "head": h,
                                    "relation": "co_occurs_with",
                                    "tail": t,
                                    "source_doc": doc_id,
                                    "page": page_no,
                                    "block_id": block_id,
                                    "evidence": text[:300],
                                }
                            )

        entities = list(entity_map.values())
        kg_index = {
            "doc_entity_index": {k: sorted(list(v)) for k, v in doc_entity_index.items()},
            "page_entity_index": {k: sorted(list(v)) for k, v in page_entity_index.items()},
            "entity_count": len(entities),
            "triple_count": len(triples),
        }
        return entities, triples, kg_index

    def _extract_entities(self, text: str, doc_id: str, page_no: int, block_id: str) -> List[Dict]:
        found = set()
        for kw in self.ENTITY_KEYWORDS:
            if kw in text:
                found.add(("keyword", kw))
        for m in self.NUMBER_PATTERN.finditer(text):
            value = m.group(1)
            if len(value) >= 2:
                found.add(("number", value))

        out = []
        for etype, value in sorted(found):
            entity_id = f"{etype}:{value}"
            out.append(
                {
                    "entity_id": entity_id,
                    "name": value,
                    "entity_type": etype,
                    "mentions": [
                        {
                            "source_doc": doc_id,
                            "page": page_no,
                            "block_id": block_id,
                        }
                    ],
                }
            )
        return out


def build_kg_from_kb(kb_path: str, output_dir: str) -> Dict[str, str]:
    kb_file = Path(kb_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(kb_file.read_text(encoding="utf-8"))

    extractor = KGExtractor()
    entities, triples, kg_index = extractor.extract_from_kb(data)

    entities_path = out_dir / "kg_entities.json"
    triples_path = out_dir / "kg_triples.json"
    index_path = out_dir / "kg_index.json"

    entities_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    triples_path.write_text(json.dumps(triples, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path.write_text(json.dumps(kg_index, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "entities_path": str(entities_path),
        "triples_path": str(triples_path),
        "index_path": str(index_path),
    }
