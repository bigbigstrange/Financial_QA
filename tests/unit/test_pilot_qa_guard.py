"""Named-App evidence and verification must fail closed."""

import pytest

from src.agent.graph import build_graph, run_agent
from src.agent.nodes import AgentConfig, AgentNodes
from src.agent.state import initial_state
from src.system.agent import QueryPlanner
from src.system.agent import GenerativeAnswerer
from src.system.pilot_budget import PilotBudgetStopped


QUESTION = "请详细说明掌上银行中基金功能的服务内容以及基金认购流程"


def test_truncated_generation_cannot_be_verified_even_if_model_would_approve():
    class ApprovingLLM:
        enabled = True
        def chat(self, *args, **kwargs):
            raise AssertionError("truncated output must be rejected before a paid check")
    nodes = AgentNodes(None, llm=ApprovingLLM(), log=lambda _: None)
    state = initial_state("说明流程")
    state.update(generation="未完成的答案", generation_truncated=True,
                 relevant_documents=[{"content": "流程说明"}],
                 answer_verified=True, answers_question=True)
    checked = nodes.check_hallucination(state)
    assert checked["grounded"] is False
    assert checked["verified"] is False
    assert checked["retry_generate"] == 1


def test_generate_sets_truncation_flag_from_transport_response():
    from types import SimpleNamespace
    class TruncatedLLM:
        enabled = True
        last_response = SimpleNamespace(finish_reason="length")
        def chat(self, *_args, **_kwargs):
            return "部分回答 [1]"
    nodes = AgentNodes(None, llm=TruncatedLLM(), log=lambda _: None)
    state = initial_state("流程是什么？")
    state["relevant_documents"] = [{"source_doc": "manual.pdf", "content": "流程", "page": 1}]
    assert nodes.generate(state)["generation_truncated"] is True


class IdentityReranker:
    def rerank(self, question, documents, top_k):
        return documents[:top_k]


class BrokenLLM:
    enabled = True

    def chat(self, *args, **kwargs):
        raise RuntimeError("offline")


class BudgetStopLLM:
    enabled = True

    def chat(self, *args, **kwargs):
        raise PilotBudgetStopped("试跑已暂停")


def test_named_app_survives_every_subquery():
    assert all("掌上银行" in q for q in QueryPlanner().plan(QUESTION))


def test_named_app_rejects_other_bank_and_generic_fund_evidence():
    nodes = AgentNodes(None, llm=None, reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state(QUESTION)
    state["documents"] = [
        {"chunk_id": "bank", "source_doc": "交通银行企业网银手册", "content": "基金认购操作流程"},
        {"chunk_id": "generic", "source_doc": "理财培训.ppt", "content": "基金认购定义与原则"},
    ]
    result = nodes.grade_documents(state)
    assert result["relevant_documents"] == []


def test_explicit_bank_rejects_other_bank_source_even_if_body_mentions_target():
    nodes = AgentNodes(None, llm=None, reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state("中国邮政储蓄银行借记卡客服电话是什么？")
    state["documents"] = [{"chunk_id": "other", "source_doc": "交通银行业务手册.pdf",
                           "content": "比较邮政储蓄银行与交通银行客服电话"}]
    assert nodes.grade_documents(state)["relevant_documents"] == []


def test_named_app_question_with_wrong_bank_results_in_no_evidence():
    class WrongBankRetriever:
        def search(self, query, top_k):
            return [{"chunk_id": "bank", "source_doc": "交通银行企业网银手册",
                     "content": "基金认购操作流程", "fusion_score": 0.9}]

    graph = build_graph(WrongBankRetriever(), llm=None, reranker=IdentityReranker(),
                        config=AgentConfig(max_retry_retrieve=0), log=lambda _: None)
    state = run_agent(graph, QUESTION)
    assert state["status"] == "no_evidence"
    assert "交通银行" not in state["generation"]


def test_second_check_failure_does_not_inherit_first_verified_flag():
    nodes = AgentNodes(None, llm=BrokenLLM(), reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state("基金认购")
    state.update({"generation": "答案", "grounded_verified": True,
                  "documents": [{"chunk_id": "c1", "content": "基金认购"}]})
    result = nodes.check_answer(state)
    assert result["verified"] is False
    assert result["answer_verified"] is False


def test_local_mode_trace_does_not_claim_both_checks_passed():
    nodes = AgentNodes(None, llm=None, reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state("基金认购")
    state.update({"generation": "候选原文", "use_paid_model": False,
                  "relevant_documents": [{"chunk_id": "c1", "content": "基金认购"}]})
    first = nodes.check_hallucination(state)
    state.update(first)
    second = nodes.check_answer(state)
    assert first["trace"][0]["decision"] == "未验证"
    assert second["trace"][0]["decision"] == "未验证"
    assert second["verified"] is False
    state.update(second)
    assert nodes.finalize(state)["trace"][0]["decision"] == "未经模型校验"


def test_negative_answer_check_is_not_called_verified_success():
    class NegativeLLM:
        enabled = True
        def chat(self, *args, **kwargs):
            return '{"answers_question": false, "reason": "偏题"}'

    nodes = AgentNodes(None, llm=NegativeLLM(), reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state("基金认购")
    state.update({"generation": "错误答案", "grounded": True,
                  "grounded_verified": True,
                  "relevant_documents": [{"chunk_id": "c1", "content": "基金认购"}]})
    result = nodes.check_answer(state)
    assert result["answer_verified"] is True  # the check ran
    assert result["answers_question"] is False
    assert result["verified"] is False  # it did not pass


def test_ark_route_is_labeled_in_usage_ledger_call():
    class CaptureArk:
        enabled = True
        provider = "ark"
        def __init__(self):
            self.nodes = []
        def chat(self, *args, **kwargs):
            self.nodes.append(kwargs["node"])
            return '{"route": "retrieval", "reason": "需要查库"}'

    llm = CaptureArk()
    AgentNodes(None, llm=llm, reranker=IdentityReranker(), log=lambda _: None).route_query(
        initial_state("基金认购"))
    assert llm.nodes == ["route_query"]


def test_legacy_answerer_does_not_hide_budget_stop():
    answerer = GenerativeAnswerer.__new__(GenerativeAnswerer)
    answerer.available = True
    answerer.backend = "api"
    answerer._dedupe_evidences = lambda rows, limit: rows
    answerer._filter_evidences_by_org = lambda rows, question: rows
    answerer._org_keys_from_question = lambda question: []
    answerer._extract_factual_answer = lambda question, rows: None
    answerer._extractive_answer = lambda question, rows, subqs: {"answer": "摘录"}
    answerer._chat_messages = lambda question, rows, subqs: []
    answerer._build_prompt = lambda question, rows, subqs: "prompt"
    def stop(*args, **kwargs):
        raise PilotBudgetStopped("试跑已暂停")
    answerer._generate_once = stop
    with pytest.raises(PilotBudgetStopped):
        answerer.generate("基金认购流程", [{"content": "基金认购流程", "source_doc": "基金.pdf"}])


@pytest.mark.parametrize("method", ["route_query", "generate", "check_hallucination", "check_answer"])
def test_budget_stop_is_not_swallowed_by_agent_nodes(method):
    nodes = AgentNodes(None, llm=BudgetStopLLM(), reranker=IdentityReranker(), log=lambda _: None)
    state = initial_state("基金认购")
    state.update({"generation": "答案", "relevant_documents": [{"chunk_id": "c1", "content": "基金认购"}]})
    with pytest.raises(PilotBudgetStopped):
        getattr(nodes, method)(state)
