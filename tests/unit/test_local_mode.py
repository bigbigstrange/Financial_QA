import os
import unittest
from unittest.mock import patch
from src.agent.nodes import AgentNodes
from app.services.qa_runtime import QARuntime


class LocalModeTests(unittest.TestCase):
    def test_text_only_validator_cannot_verify_pure_image(self):
        from types import SimpleNamespace
        image = {'block_id': 'vp', 'content': '', 'evidence_kind': 'page_image'}
        node = AgentNodes(None, llm=SimpleNamespace(enabled=True), log=lambda _: None)
        node._chat_json = lambda *a, **k: {'grounded': True, 'answers_question': True}
        state = {'question': '问题', 'documents': [image], 'generation': '仅图像证据'}
        self.assertFalse(node.check_hallucination(state)['verified'])
        self.assertFalse(node.check_answer(state)['verified'])

    def test_generation_prompt_labels_image_as_unread(self):
        from src.agent.prompts import build_generate_prompt
        prompt = build_generate_prompt('问题', [{'evidence_kind': 'page_image', 'content': '', 'page': 1}])
        self.assertIn('未经文字解析', prompt)

    def test_legacy_pure_image_answer_does_not_report_no_evidence(self):
        from src.system.agent import GenerativeAnswerer
        answerer = GenerativeAnswerer.__new__(GenerativeAnswerer)
        answer = answerer.generate('数值是多少？', [{'block_id': 'vp', 'content': '',
            'evidence_kind': 'page_image', 'source_doc': '图', 'page': 1}])
        self.assertIn('仅图像证据', answer['answer'])

    def test_pure_image_extraction_is_explicit_and_does_not_invent_text(self):
        from src.agent import build_graph, run_agent
        class Retriever:
            def search(self, query, top_k=12):
                return [{'block_id': 'vp_1', 'chunk_id': None, 'content': '',
                         'evidence_kind': 'page_image', 'source_doc': '图.png', 'page': 2}]
        result = run_agent(build_graph(retriever=Retriever(), llm=None), '图里的数值是多少？')
        self.assertIn('仅图像证据', result['generation'])
        self.assertIn('未经文字解析', result['generation'])
        self.assertFalse(result.get('verified'))
        self.assertEqual(result['documents'][0]['content'], '')

    def test_excerpt_does_not_invent_page_or_claim_verified_relevance(self):
        answer = AgentNodes._extractive_answer([{"content": "原文", "source_doc": "a.txt", "page": None}])
        self.assertIn("尚未校验", answer)
        self.assertIn("无物理页码", answer)
        self.assertNotIn("None页", answer)

    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "1"})
    def test_runtime_marks_local_results_unverified(self):
        runtime = QARuntime()
        with patch.object(runtime, "_build_agent_graph", return_value=object()), patch("src.agent.run_agent", return_value={
            "generation": "摘录", "confidence": 1.0, "trace": [], "documents": []}):
            result = runtime._answer_with_graph("问题", "run")
        self.assertLessEqual(result["confidence"], 0.3)
        self.assertIn("未经", result["answer"])
        self.assertEqual(result["trace"][0]["decision"], "local_retrieval_only")

    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "0"})
    def test_per_question_paid_switch_keeps_result_local_and_unverified(self):
        runtime = QARuntime()
        state = {"generation": "摘录", "confidence": 1.0, "trace": [],
                 "documents": [], "use_paid_model": False, "use_web_search": False}
        with patch.object(runtime, "_build_agent_graph", return_value=object()), \
             patch("src.agent.run_agent", return_value=state) as run:
            result = runtime._answer_with_graph("问题", "run", use_paid_model=False,
                                                use_web_search=False)
        self.assertFalse(run.call_args.kwargs["use_paid_model"])
        self.assertFalse(run.call_args.kwargs["use_web_search"])
        self.assertIn("本次已关闭豆包模型", result["answer"])
        self.assertFalse(result["verified"])
        self.assertTrue(result["_local_only"])

    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "1"})
    def test_rejected_wrong_bank_hits_are_not_persisted_as_answer_evidence(self):
        runtime = QARuntime()
        wrong = {"chunk_id": "wrong", "content": "交通银行企业网银基金认购",
                 "source_doc": "交通银行企业网银手册"}
        state = {"generation": "知识库中无依据", "status": "no_evidence",
                 "documents": [wrong], "relevant_documents": [], "trace": []}
        with patch.object(runtime, "_build_agent_graph", return_value=object()), \
             patch("src.agent.run_agent", return_value=state):
            result = runtime._answer_with_graph("掌上银行基金认购", "run")
        self.assertEqual(result["evidence"], [])
