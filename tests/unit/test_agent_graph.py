"""Agent（LangGraph）图结构与条件边契约测试。

用假的 LLM / 检索器 / 重排器验证五件事：
  1. 图能编译，快乐路径按顺序走完全部节点；
  2. 路由到 direct / clarify 时短路，不进入检索；
  3. **幻觉分支回 generate**（同批证据重生成），`retry_generate` 递增；
  4. **偏题分支回 retrieve**（换信息源），`retry_retrieve` 递增；
  5. 无 LLM 时降级可跑通，且重试有硬上限、不会死循环。
"""
import json
import re
import unittest

from src.agent import build_graph, run_agent
from src.agent.nodes import AgentConfig, _supports_named_app
from src.agent.state import AgentState


class FakeLLM:
    enabled = True

    def __init__(self, rules: list[tuple[str, str]]):
        self.rules = rules
        self.calls: list[str] = []

    def chat(self, messages, temperature=0.2, max_tokens=1024):
        system = messages[0]["content"]
        self.calls.append(system)
        for keyword, response in self.rules:
            if keyword in system:
                if response == GRADE_ALL_RELEVANT:
                    count = len(re.findall(r"(?m)^\[\d+\]", messages[1]["content"]))
                    return json.dumps({"grades": [{"index": i, "relevant": True}
                                                   for i in range(1, count + 1)]})
                return response
        return "{}"


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.queries: list[str] = []

    def search(self, query, top_k=10, visibility=None):
        self.queries.append(query)
        return [dict(d) for d in self.docs[:top_k]]


class FakeReranker:
    def __init__(self, order=None):
        self.order = order

    def rerank(self, question, evidences, top_k, sub_questions=None):
        items = list(evidences)
        if self.order:
            items.sort(key=lambda x: self.order.index(x["chunk_id"])
                       if x["chunk_id"] in self.order else 99)
        return items[:top_k]


def _doc(cid: str, content: str = "营业收入1000亿元，同比增长5%。") -> dict:
    return {
        "chunk_id": cid,
        "block_id": cid,
        "content": content,
        "source_doc": "某银行2023年报.pdf",
        "page_start": 3,
        "page_end": 3,
        "section_path": "经营概况",
        "type": "paragraph",
        "fusion_score": 0.5,
        "retrieval_channels": ["dense"],
    }


ROUTE_RETRIEVAL = '{"route": "retrieval", "reason": "需要查业务规则"}'
ROUTE_DIRECT = '{"route": "direct", "reason": "寒暄"}'
ROUTE_CLARIFY = '{"route": "clarify", "reason": "缺少业务对象"}'
GRADE_ALL_RELEVANT = '{"grades": [{"index": 1, "relevant": true, "reason": "相关"}]}'
GRADE_NONE = '{"grades": [{"index": 1, "relevant": false, "reason": "无关"}]}'
GROUNDED_OK = '{"grounded": true, "reason": "有据可依"}'
GROUNDED_BAD = '{"grounded": false, "reason": "出现了证据里没有的数字"}'
ANSWER_OK = '{"answers_question": true, "reason": "回答了"}'
ANSWER_BAD = '{"answers_question": false, "reason": "答非所问"}'


def _nodes(state: AgentState) -> list[str]:
    return [t["node"] for t in state.get("trace", [])]


class HappyPathTests(unittest.TestCase):
    def test_recursion_limit_is_explicit_and_preserves_thread_config(self):
        class RecordingGraph:
            def invoke(self, state, config):
                return {"state": state, "config": config}

        result = run_agent(RecordingGraph(), "问题", run_id="run-1",
                           config={"tags": ["audit"]})
        self.assertEqual(result["config"]["recursion_limit"], 32)
        self.assertEqual(result["config"]["configurable"]["thread_id"], "run-1")
        self.assertEqual(result["config"]["tags"], ["audit"])

    def test_eight_langgraph_supersteps_cannot_finish_normal_eight_node_path(self):
        from langgraph.errors import GraphRecursionError

        llm = FakeLLM([
            ("路由分派器", ROUTE_RETRIEVAL),
            ("检索质量评审员", GRADE_ALL_RELEVANT),
            ("事实一致性校验器", GROUNDED_OK),
            ("答案完整性校验器", ANSWER_OK),
        ])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            planner=_FakePlanner(), reranker=FakeReranker())
        with self.assertRaises(GraphRecursionError):
            run_agent(graph, "某银行收入？", config={"recursion_limit": 8})
        self.assertEqual(run_agent(graph, "某银行收入？")["status"], "completed")

    def test_full_path_visits_expected_nodes(self):
        llm = FakeLLM([
            ("路由分派器", ROUTE_RETRIEVAL),
            ("检索质量评审员", GRADE_ALL_RELEVANT),
            ("事实一致性校验器", GROUNDED_OK),
            ("答案完整性校验器", ANSWER_OK),
        ])
        graph = build_graph(FakeRetriever([_doc("ck_1"), _doc("ck_2")]), llm,
                            config=AgentConfig(max_retry_retrieve=2),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "某银行2023年营业收入是多少？")
        visited = _nodes(state)
        for expected in ("route_query", "plan_subquestions", "retrieve",
                         "grade_documents", "generate", "check_hallucination",
                         "check_answer", "finalize"):
            self.assertIn(expected, visited)
        self.assertNotIn("rewrite_query", visited)
        self.assertNotIn("give_up", visited)
        # 收尾字段必须在成功路径也被写入（历史上漏过，调用方拿到 None）
        self.assertEqual(state.get("status"), "completed")
        self.assertIsNotNone(state.get("confidence"))
        self.assertGreater(state.get("confidence"), 0.0)

    def test_trace_records_timing_and_decisions(self):
        llm = FakeLLM([
            ("路由分派器", ROUTE_RETRIEVAL),
            ("检索质量评审员", GRADE_ALL_RELEVANT),
            ("事实一致性校验器", GROUNDED_OK),
            ("答案完整性校验器", ANSWER_OK),
        ])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        for entry in state["trace"]:
            self.assertIn("node", entry)
            self.assertIsInstance(entry.get("elapsed_ms"), int)

    def test_retrieval_merges_subquestions_without_duplicates(self):
        """多子问题检索后必须按 chunk_id 去重。"""
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_OK)])
        retriever = FakeRetriever([_doc("ck_1"), _doc("ck_2")])
        graph = build_graph(retriever, llm, planner=_FakePlanner(2),
                            reranker=FakeReranker())
        state = run_agent(graph, "问题")
        ids = [d["chunk_id"] for d in state["relevant_documents"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(retriever.queries), 2)


class RoutingTests(unittest.TestCase):
    def test_direct_route_short_circuits_retrieval(self):
        llm = FakeLLM([("路由分派器", ROUTE_DIRECT)])
        retriever = FakeRetriever([_doc("ck_1")])
        graph = build_graph(retriever, llm, planner=_FakePlanner(),
                            reranker=FakeReranker())
        state = run_agent(graph, "你好")
        self.assertEqual(state.get("status"), "direct")
        self.assertEqual(retriever.queries, [])
        self.assertEqual(_nodes(state), ["route_query", "direct_answer"])

    def test_clarify_route_short_circuits_retrieval(self):
        llm = FakeLLM([("路由分派器", ROUTE_CLARIFY)])
        retriever = FakeRetriever([_doc("ck_1")])
        graph = build_graph(retriever, llm, planner=_FakePlanner(),
                            reranker=FakeReranker())
        state = run_agent(graph, "怎么办")
        self.assertEqual(state.get("status"), "clarify")
        self.assertEqual(retriever.queries, [])

    def test_invalid_route_response_falls_back_to_retrieval(self):
        llm = FakeLLM([("路由分派器", '{"route": "nonsense"}'),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_OK)])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(state.get("route"), "retrieval")


class DoubleCheckTests(unittest.TestCase):
    def test_hallucination_retries_generate_not_retrieve(self):
        """幻觉 → 回 generate（同批证据重生成），且检索只发生一次。"""
        llm = FakeLLM([
            ("路由分派器", ROUTE_RETRIEVAL),
            ("检索质量评审员", GRADE_ALL_RELEVANT),
            ("事实一致性校验器", GROUNDED_BAD),
            ("答案完整性校验器", ANSWER_OK),
        ])
        retriever = FakeRetriever([_doc("ck_1")])
        graph = build_graph(retriever, llm,
                            config=AgentConfig(max_retry_generate=1),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        visited = _nodes(state)
        self.assertEqual(visited.count("retrieve"), 1)      # 没有重新检索
        self.assertEqual(visited.count("generate"), 2)      # 生成被重试了一次
        # retry_generate 记录「校验不通过次数」，1 次重试后为 2
        self.assertEqual(state.get("retry_generate"), 2)
        self.assertEqual(state.get("retry_retrieve"), 0)
        self.assertEqual(state.get("status"), "no_evidence")

    def test_hallucination_gives_up_after_limit(self):
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_BAD),
                       ("答案完整性校验器", ANSWER_OK)])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            config=AgentConfig(max_retry_generate=0),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(state.get("status"), "no_evidence")
        self.assertIn("give_up", _nodes(state))

    def test_answer_mismatch_re_retrieves(self):
        """偏题 → 回 retrieve（换信息源），并触发查询改写。"""
        llm = FakeLLM([
            ("路由分派器", ROUTE_RETRIEVAL),
            ("检索质量评审员", GRADE_ALL_RELEVANT),
            ("事实一致性校验器", GROUNDED_OK),
            ("答案完整性校验器", ANSWER_BAD),
            ("查询改写器", '{"query": "改写后的查询"}'),
        ])
        retriever = FakeRetriever([_doc("ck_1")])
        graph = build_graph(retriever, llm,
                            config=AgentConfig(max_retry_retrieve=1),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        visited = _nodes(state)
        self.assertEqual(visited.count("retrieve"), 2)
        self.assertEqual(visited.count("rewrite_query"), 1)
        self.assertEqual(state.get("retry_retrieve"), 1)
        # 重试时扩大了召回窗口
        self.assertGreater(state.get("retrieve_top_k"), 12)
        # 最终仍偏题且已达上限 → 放弃而不是死循环
        self.assertEqual(state.get("status"), "no_evidence")

    def test_answer_mismatch_never_loops_forever(self):
        """回归测试：偏题分支曾因绕过 rewrite_query 导致无限递归。"""
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_BAD),
                       ("查询改写器", '{"query": "改写"}')])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            config=AgentConfig(max_retry_retrieve=3),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        visited = _nodes(state)
        self.assertLessEqual(visited.count("retrieve"), 4)
        self.assertEqual(state.get("retry_retrieve"), 3)
        self.assertEqual(state.get("status"), "no_evidence")

    def test_answer_mismatch_gives_up_after_limit(self):
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_BAD),
                       ("查询改写器", '{"query": "改写"}')])
        graph = build_graph(FakeRetriever([_doc("ck_1")]), llm,
                            config=AgentConfig(max_retry_retrieve=0),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(state.get("status"), "no_evidence")

    def test_no_relevant_evidence_triggers_rewrite(self):
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_NONE),
                       ("查询改写器", '{"query": "改写"}'),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_OK)])
        graph = build_graph(FakeRetriever([]), llm,
                            config=AgentConfig(max_retry_retrieve=1),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertIn("rewrite_query", _nodes(state))
        self.assertEqual(state.get("relevant_documents"), [])

    def test_empty_knowledge_gives_up_without_looping(self):
        graph = build_graph(FakeRetriever([]), None,
                            config=AgentConfig(max_retry_retrieve=0),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(state.get("status"), "no_evidence")
        self.assertIn("知识库中无依据", state.get("generation") or "")


class DegradedModeTests(unittest.TestCase):
    def test_runs_without_llm_and_uses_extractive_answer(self):
        graph = build_graph(FakeRetriever([_doc("ck_1")]), None,
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "某银行营业收入是多少？")
        self.assertEqual(state.get("route"), "retrieval")
        self.assertIn("营业收入", state.get("generation") or "")
        self.assertEqual(state.get("grounded"), True)
        self.assertEqual(state.get("answers_question"), True)

    def test_llm_exception_does_not_break_graph(self):
        class ExplodingLLM:
            enabled = True

            def chat(self, *a, **kw):
                raise RuntimeError("模拟 API 故障")

        graph = build_graph(FakeRetriever([_doc("ck_1")]), ExplodingLLM(),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertTrue(state.get("generation"))
        self.assertEqual(state.get("route"), "retrieval")

    def test_retriever_exception_is_tolerated(self):
        class ExplodingRetriever:
            def search(self, *a, **kw):
                raise RuntimeError("模拟检索故障")

        graph = build_graph(ExplodingRetriever(), None,
                            config=AgentConfig(max_retry_retrieve=0),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(state.get("status"), "no_evidence")


class _FakePlanner:
    def __init__(self, count: int = 1):
        self.count = count

    def plan(self, question: str):
        if self.count <= 1:
            return [question]
        return [f"{question} 子问题{i + 1}" for i in range(self.count)]


class ConfidenceTests(unittest.TestCase):
    """置信度必须可解释且有区分度 —— 恒定值等于没有信息量。"""

    def _run(self, llm, docs):
        graph = build_graph(FakeRetriever(docs), llm, planner=_FakePlanner(),
                            reranker=FakeReranker())
        return run_agent(graph, "问题")

    def test_confidence_scales_with_evidence_count(self):
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_OK)])
        few = self._run(llm, [_doc("ck_1")])
        many = self._run(llm, [_doc(f"ck_{i}") for i in range(12)])
        self.assertLess(few["confidence"], many["confidence"])
        self.assertEqual(many["confidence"], 1.0)
        self.assertLess(few["confidence"], 0.5)

    def test_unverified_run_is_discounted(self):
        """无 LLM 时双层校验是空过，不能给满分置信度。"""
        no_llm = self._run(None, [_doc("ck_1")])
        self.assertFalse(no_llm.get("verified"))
        with_llm = self._run(
            FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                     ("检索质量评审员", GRADE_ALL_RELEVANT),
                     ("事实一致性校验器", GROUNDED_OK),
                     ("答案完整性校验器", ANSWER_OK)]),
            [_doc("ck_1")],
        )
        self.assertTrue(with_llm.get("verified"))
        self.assertLess(no_llm["confidence"], with_llm["confidence"])

    def test_failed_checks_halve_confidence(self):
        good = self._run(
            FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                     ("检索质量评审员", GRADE_ALL_RELEVANT),
                     ("事实一致性校验器", GROUNDED_OK),
                     ("答案完整性校验器", ANSWER_OK)]),
            [_doc("ck_1")],
        )
        bad = self._run(
            FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                     ("检索质量评审员", GRADE_ALL_RELEVANT),
                     ("事实一致性校验器", GROUNDED_BAD),
                     ("答案完整性校验器", ANSWER_OK)]),
            [_doc("ck_1")],
        )
        self.assertLess(bad["confidence"], good["confidence"])

    def test_grade_keeps_full_document_list(self):
        """documents 应保持完整召回集，不能缩成评分候选子集。"""
        llm = FakeLLM([("路由分派器", ROUTE_RETRIEVAL),
                       ("检索质量评审员", GRADE_ALL_RELEVANT),
                       ("事实一致性校验器", GROUNDED_OK),
                       ("答案完整性校验器", ANSWER_OK)])
        docs = [_doc(f"ck_{i}") for i in range(12)]
        graph = build_graph(FakeRetriever(docs), llm,
                            config=AgentConfig(max_grade_candidates=10),
                            planner=_FakePlanner(), reranker=FakeReranker())
        state = run_agent(graph, "问题")
        self.assertEqual(len(state["documents"]), 12)
        self.assertEqual(state["graded_count"], 10)
        self.assertEqual(len(state["relevant_documents"]), 10)


def test_missing_or_non_boolean_grades_never_become_answer_evidence():
    grades = {"grades": [{"index": 1, "relevant": True},
                         {"index": 2, "relevant": "false"}]}
    llm = FakeLLM([("检索质量评审员", json.dumps(grades))])
    from src.agent.nodes import AgentNodes

    nodes = AgentNodes(None, llm, reranker=FakeReranker(), log=lambda _: None)
    state = {"question": "营业收入？", "documents": [_doc("one"), _doc("two"), _doc("three")],
             "use_paid_model": True, "web_searched": False}
    result = nodes.grade_documents(state)
    assert [item["chunk_id"] for item in result["relevant_documents"]] == ["one"]
    assert nodes.generate({**state, "relevant_documents": []})["generation"].startswith("知识库中无依据")


def test_duplicate_grades_are_not_accepted():
    grades = {"grades": [{"index": 1, "relevant": True},
                         {"index": 1, "relevant": False}]}
    llm = FakeLLM([("检索质量评审员", json.dumps(grades))])
    from src.agent.nodes import AgentNodes

    nodes = AgentNodes(None, llm, reranker=FakeReranker(), log=lambda _: None)
    result = nodes.grade_documents({"question": "营业收入？", "documents": [_doc("one")],
                                    "use_paid_model": True, "web_searched": False})
    assert result["relevant_documents"] == []


def test_named_bank_rejects_generic_or_other_institution_evidence():
    question = "赣昌农商银行征信业务管理暂行办法中征信业务有哪些？"
    assert _supports_named_app(question, {
        "source_doc": "赣昌农商银行征信业务管理暂行办法.doc",
        "content": "赣昌农商银行征信业务包括数据采集。",
    })
    assert not _supports_named_app(question, {
        "source_doc": "商业银行保理业务管理暂行办法.docx",
        "content": "商业银行办理保理业务。",
    })


if __name__ == "__main__":
    unittest.main()
