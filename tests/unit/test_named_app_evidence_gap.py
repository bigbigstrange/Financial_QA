"""An App procedure question must not be answered with unrelated fund manuals."""

from src.agent.graph import build_graph, run_agent
from src.agent.nodes import AgentConfig, AgentNodes
from src.agent.state import initial_state


QUESTION = "请详细说明掌上银行中基金功能的服务内容以及基金认购的具体操作流程"

APP_MARKETING = {
    "chunk_id": "app_marketing",
    "block_id": "app_marketing",
    "source_doc": "农业银行分析报告.pdf",
    "content": "农业银行推出个人掌上银行，手机银行月活客户数持续增长。",
    "fusion_score": 0.9,
}
OTHER_BANK = {
    "chunk_id": "other_bank",
    "block_id": "other_bank",
    "source_doc": "交通银行业务手册.docx",
    "content": "客户登录网银后选择基金认购，确认交易。",
    "fusion_score": 0.8,
}
GENERIC_FUND = {
    "chunk_id": "generic_fund",
    "block_id": "generic_fund",
    "source_doc": "银行从业资格培训.ppt",
    "content": "基金认购是在募集期购买基金份额的行为。",
    "fusion_score": 0.7,
}


def test_app_marketing_and_other_bank_fund_steps_are_insufficient():
    state = initial_state(QUESTION)
    state["documents"] = [APP_MARKETING, OTHER_BANK, GENERIC_FUND]

    graded = AgentNodes(None, llm=None).grade_documents(state)

    assert graded["relevant_documents"] == []


def test_graph_refuses_to_invent_missing_app_procedure():
    class CorpusShapedRetriever:
        def search(self, query, top_k):
            return [APP_MARKETING, OTHER_BANK, GENERIC_FUND][:top_k]

    graph = build_graph(
        CorpusShapedRetriever(),
        llm=None,
        config=AgentConfig(max_retry_retrieve=0),
    )

    result = run_agent(graph, QUESTION)

    assert result["status"] == "no_evidence"
    assert "确认交易" not in result["generation"]


def test_matching_app_and_fund_manual_still_remains_eligible():
    state = initial_state(QUESTION)
    state["documents"] = [{
        "chunk_id": "matching",
        "block_id": "matching",
        "source_doc": "掌上银行基金操作手册.pdf",
        "content": "掌上银行基金认购：选择基金，阅读风险提示，输入认购金额并确认。",
        "fusion_score": 0.8,
    }]

    graded = AgentNodes(None, llm=None).grade_documents(state)

    assert [row["chunk_id"] for row in graded["relevant_documents"]] == ["matching"]


def test_marketing_with_both_app_and_fund_words_is_not_procedure_evidence():
    state = initial_state(QUESTION)
    state["documents"] = [{
        "chunk_id": "marketing",
        "block_id": "marketing",
        "source_doc": "农业银行分析报告.pdf",
        "content": "掌上银行客户增长，基金业务稳步发展。",
        "fusion_score": 0.9,
    }]

    graded = AgentNodes(None, llm=None).grade_documents(state)

    assert graded["relevant_documents"] == []


def test_image_only_app_page_remains_unverified_candidate():
    state = initial_state(QUESTION)
    state["documents"] = [{
        "block_id": "page_one",
        "page_id": "page_one",
        "evidence_kind": "page_image",
        "source_doc": "产品手册.pdf",
        "image_ref": "page_one.png",
        "content": "",
        "fusion_score": 0.8,
    }]

    graded = AgentNodes(None, llm=None).grade_documents(state)

    assert [row["block_id"] for row in graded["relevant_documents"]] == ["page_one"]
