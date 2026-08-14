import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .agent import DocumentAgent
from cache_paths import resolve_cache_dir
from .model_settings import resolve_embed_model, resolve_gen_model
from .observability import log_event
from .retrieval import build_retrieval_stack


class FinancialQASystem:
    def __init__(
        self,
        mmd_directory,
        model_name="Qwen2.5-VL",
        embed_model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        image_backend: str = "visrag",
        enable_graph=True,
        kg_dir="./output",
        reranker_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = resolve_cache_dir(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.kb_path = self._resolve_kb_path(mmd_directory)
        embed_model = embed_model if embed_model is not None else resolve_embed_model()
        gen_model = gen_model if gen_model is not None else resolve_gen_model()
        self.text_model = embed_model
        self.embed_model = embed_model
        self.image_model = embed_model
        self.image_backend = image_backend
        self.enable_graph = enable_graph
        self.kg_dir = kg_dir

        self.store = None
        self.text_retriever = None
        self.image_retriever = None
        self.hybrid_retriever = None
        # 延迟创建：避免在仅构建/加载向量索引时拉取 Qwen 等生成模型
        self._agent: Optional[DocumentAgent] = None
        self._gen_model = gen_model
        self.knowledge_chunks: List[Dict] = []
        self.vector_index = None
        _ = reranker_model  # 兼容旧参数；重排已改为 heuristic/hybrid/api

    def _get_agent(self) -> DocumentAgent:
        if self._agent is None:
            self._agent = DocumentAgent(gen_model=self._gen_model)
        return self._agent

    @staticmethod
    def _resolve_kb_path(input_path: str) -> str:
        p = Path(input_path)
        if p.suffix.lower() == ".json":
            return str(p)
        if p.is_dir():
            candidate = p / "knowledge_base.json"
            if candidate.exists():
                return str(candidate)
        if p.suffix.lower() == ".md":
            candidate = p.parent / "knowledge_base.json"
            if candidate.exists():
                return str(candidate)
        fallback = Path("./output/knowledge_base.json")
        return str(fallback)

    def build_vector_index(self, force_rebuild: bool = False):
        self.store, self.text_retriever, self.image_retriever, self.hybrid_retriever = build_retrieval_stack(
            kb_path=self.kb_path,
            cache_dir=str(self.cache_dir),
            text_model=self.text_model,
            image_proxy_model=self.image_model,
            image_backend=self.image_backend,
            enable_graph=self.enable_graph,
            kg_dir=self.kg_dir,
            neo4j_uri=os.getenv("NEO4J_URI", ""),
            neo4j_user=os.getenv("NEO4J_USER", ""),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            force_rebuild=force_rebuild,
        )
        self.knowledge_chunks = self.store.blocks
        self.vector_index = self.text_retriever.index
        log_event(
            "index_built",
            chunks=len(self.knowledge_chunks),
            kb_path=self.kb_path,
            image_backend=self.image_backend,
        )

    def search_similar_chunks(self, query: str, top_k: int = 10) -> List[Dict]:
        if self.hybrid_retriever is None:
            self.build_vector_index()
        return self.hybrid_retriever.search(query, top_k=top_k)

    def answer_question(self, question: str, question_id: str) -> Dict:
        if self.hybrid_retriever is None:
            self.build_vector_index()

        # 阶段4：统一走 reranker + 生成式回答链路
        result = self._get_agent().run(question, retriever=self.hybrid_retriever, top_k=12)

        return {
            "question_id": question_id,
            "question": question,
            "answer": result.get("answer", ""),
            "thinking": result.get("thinking", ""),
            "sub_questions": result.get("sub_questions", []),
            "knowledge_points": [p["text"] for p in result["knowledge_points"]],
            "evidence": result["knowledge_points"],
            "confidence": result["confidence"],
        }

    def process_test_set(self, test_set_path: str) -> List[Dict]:
        df = pd.read_excel(test_set_path)
        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="处理测试问题"):
            qid = str(row["question_id"])
            q = str(row["question"])
            try:
                rows.append(self.answer_question(q, qid))
            except Exception as e:
                rows.append(
                    {
                        "question_id": qid,
                        "question": q,
                        "knowledge_points": ["处理失败，请重试"],
                        "evidence": [],
                        "confidence": 0.0,
                        "error": str(e),
                    }
                )
        return rows

    def save_results(self, results: List[Dict], output_path: str):
        out_json = Path(output_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        out_md = out_json.with_suffix(".md")
        lines = ["# QA Results", ""]
        for item in results:
            lines.append(f"## {item['question_id']}")
            lines.append("")
            lines.append(f"**Q:** {item.get('question', '')}")
            lines.append("")
            if item.get("answer"):
                lines.append(f"**A:** {item.get('answer', '')}")
                lines.append("")
            lines.append(f"**Confidence:** {item.get('confidence', 0.0)}")
            lines.append("")
            for idx, ev in enumerate(item.get("evidence", []), start=1):
                lines.append(
                    f"{idx}. {ev.get('text', '')}  \n"
                    f"   - source_doc: {ev.get('source_doc')}  \n"
                    f"   - page: {ev.get('page')}  \n"
                    f"   - block_id: {ev.get('block_id')}  \n"
                    f"   - score: {ev.get('score')}"
                )
            lines.append("")
        out_md.write_text("\n".join(lines), encoding="utf-8")
