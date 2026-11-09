"""Web routing and Tavily evidence stay bounded and attributable."""

import json
from unittest.mock import patch

from src.agent import build_graph, run_agent
from src.agent.nodes import AgentConfig, AgentNodes
from src.contracts import evidence_from_v2
from src.system.web_search import TavilySearch


class LLM:
    enabled = True

    def __init__(self, route="retrieval", grade=False, web_grade=True):
        self.route = route
        self.grade = grade
        self.web_grade = web_grade

    def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if "路由分派器" in system:
            return json.dumps({"route": self.route, "reason": "测试"})
        if "检索质量评审员" in system:
            is_web = "URL=https://example.com" in messages[1]["content"]
            return json.dumps({"grades": [{"index": 1,
                "relevant": self.web_grade if is_web else self.grade}]})
        if "事实一致性校验器" in system:
            return '{"grounded": true}'
        if "答案完整性校验器" in system:
            return '{"answers_question": true}'
        return "根据证据回答[1]"


class Retriever:
    def __init__(self, docs=()):
        self.docs = docs
        self.calls = 0

    def search(self, query, top_k=12):
        self.calls += 1
        return list(self.docs)


class Search:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.results


class Reranker:
    def rerank(self, question, evidences, top_k):
        return evidences[:top_k]


class Planner:
    def plan(self, question):
        return [question]


WEB = [{"chunk_id": "web:1", "source_doc": "公告", "source_url": "https://example.com/a",
        "content": "公告中的利率为 3%", "retrieval_channels": ["web"]}]
LOCAL = [{"chunk_id": "local:1", "source_doc": "旧文件", "content": "无关内容",
          "fusion_score": 0.5, "retrieval_channels": ["dense"]}]


def graph(retriever, llm, search):
    return build_graph(retriever, llm, web_search=search,
                       config=AgentConfig(max_retry_retrieve=0),
                       planner=Planner(), reranker=Reranker(), log=lambda _: None)


def test_per_question_switches_disable_model_and_web_calls():
    class CountingLLM(LLM):
        calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return super().chat(messages, **kwargs)

    llm, search = CountingLLM(route="websearch"), Search(WEB)
    state = run_agent(graph(Retriever(LOCAL), llm, search), "查询业务？",
                      use_paid_model=False, use_web_search=False)
    assert llm.calls == 0
    assert search.queries == []
    assert state["verified"] is False


def test_per_question_web_switch_overrides_model_route_and_fallback():
    search = Search(WEB)
    state = run_agent(graph(Retriever(LOCAL), LLM(route="websearch"), search),
                      "查询业务？", use_web_search=False)
    assert state["route"] == "retrieval"
    assert search.queries == []


def test_tavily_can_be_allowed_without_doubao_but_is_not_verified():
    search = Search(WEB)
    state = run_agent(graph(Retriever(), LLM(route="websearch"), search),
                      "今日利率？", use_paid_model=False, use_web_search=True)
    assert search.queries == ["今日利率？"]
    assert state["verified"] is False
    assert state["status"] == "no_evidence"


def test_direct_web_route_uses_original_question_and_citable_url():
    retriever, search = Retriever(LOCAL), Search(WEB)
    state = run_agent(graph(retriever, LLM(route="websearch"), search), "今日利率？")
    assert retriever.calls == 0
    assert search.queries == ["今日利率？"]
    assert state["status"] == "completed"
    assert state["relevant_documents"][0]["source_url"] == "https://example.com/a"
    assert state["citations"] == [1]
    assert "web_search" in [entry["node"] for entry in state["trace"]]


def test_grounding_receives_same_source_metadata_and_excerpt_as_generation():
    from src.agent.prompts import build_generate_prompt, build_hallucination_prompt

    body = "背景" * 450 + "利率为3%"
    evidence = [{**WEB[0], "content": body}]
    generate = build_generate_prompt("利率是多少？", evidence)
    grounding = build_hallucination_prompt("利率是多少？", "3%[1]", evidence)
    for fragment in ("文档=公告 URL=https://example.com/a", body):
        assert fragment in generate
        assert fragment in grounding


def test_web_route_without_search_key_falls_back_to_local():
    retriever = Retriever(LOCAL)
    with patch.dict("os.environ", {"TAVILY_API_KEY": ""}):
        state = run_agent(graph(retriever, LLM(route="websearch"), None), "今日利率？")
    assert retriever.calls == 1
    assert state["route"] == "retrieval"


def test_irrelevant_local_documents_fall_back_to_web():
    retriever, search = Retriever(LOCAL), Search(WEB)
    state = run_agent(graph(retriever, LLM(grade=False), search), "今日利率？")
    assert retriever.calls == 1
    assert search.queries == ["今日利率？"]
    assert state["documents"] == LOCAL
    assert state["relevant_documents"][0]["source_url"] == "https://example.com/a"


def test_successful_web_search_preserves_previously_graded_local_evidence():
    search = Search(WEB)
    nodes = AgentNodes(Retriever(), LLM(web_grade=True), web_search=search,
                       reranker=Reranker(), log=lambda _: None)
    local = {**LOCAL[0], "relevant": True}
    initial = {"question": "今日利率？", "relevant_documents": [local],
               "use_web_search": True, "use_paid_model": True}
    searched = nodes.web_search(initial)
    graded = nodes.grade_documents({**initial, **searched})
    assert [item["chunk_id"] for item in graded["relevant_documents"]] == ["local:1", "web:1"]
    assert graded["filtered_count"] == 0


def test_unrelated_web_snippet_is_not_used_as_answer_evidence():
    search = Search(WEB)
    state = run_agent(graph(Retriever(), LLM(route="websearch", web_grade=False), search),
                      "今日利率？")
    assert state["status"] == "no_evidence"
    assert state["relevant_documents"] == []
    assert [item["node"] for item in state["trace"]].count("grade_documents") == 1


def test_web_snippet_fails_closed_without_model_grading():
    state = run_agent(graph(Retriever(), None, Search(WEB)), "今日利率？")
    assert state["status"] == "no_evidence"
    assert state["relevant_documents"] == []


def test_web_failure_ends_without_recursion_or_stale_local_evidence():
    class BrokenSearch:
        def search(self, query):
            raise TimeoutError("offline")

    state = run_agent(graph(Retriever(LOCAL), LLM(grade=False), BrokenSearch()), "今日利率？")
    assert state["status"] == "no_evidence"
    assert state["relevant_documents"] == []
    assert [entry["node"] for entry in state["trace"]].count("web_search") == 1
    assert "网页搜索失败" in state["generation"]
    assert "超时" in state["generation"]
    assert "未找到能回答" not in state["generation"]
    assert next(entry for entry in state["trace"] if entry["node"] == "web_search")["decision"] == "search_failed"


def test_web_unauthorized_reports_key_problem_without_leaking_key():
    class UnauthorizedSearch:
        def search(self, query):
            raise RuntimeError("Tavily HTTP 401")

    state = run_agent(graph(Retriever(), LLM(route="websearch"), UnauthorizedSearch()),
                      "今日利率？")
    assert "HTTP 401" in state["generation"]
    assert "密钥" in state["generation"]
    assert "TAVILY_API_KEY=" not in state["generation"]


def test_web_rate_limit_reports_quota_instead_of_missing_evidence():
    class LimitedSearch:
        def search(self, query):
            raise RuntimeError("Tavily HTTP 429")

    state = run_agent(graph(Retriever(), LLM(route="websearch"), LimitedSearch()),
                      "今日利率？")
    assert "HTTP 429" in state["generation"]
    assert "额度" in state["generation"]
    assert "未找到能回答" not in state["generation"]


def test_web_connection_failure_mentions_possible_timeout():
    class UnavailableSearch:
        def search(self, query):
            raise RuntimeError("Tavily search unavailable")

    state = run_agent(graph(Retriever(), LLM(route="websearch"), UnavailableSearch()),
                      "今日利率？")
    assert "网页搜索失败" in state["generation"]
    assert "超时" in state["generation"]


def test_empty_successful_web_search_is_not_reported_as_service_failure():
    state = run_agent(graph(Retriever(), LLM(route="websearch"), Search([])),
                      "今日利率？")
    assert "未找到能回答" in state["generation"]
    assert "网页搜索失败" not in state["generation"]


def test_web_results_for_other_bank_cannot_answer_named_bank_question():
    wrong_bank = [{"chunk_id": "web:1", "source_doc": "建设银行公告",
                   "source_url": "https://example.com/a",
                   "content": "建设银行利率为 3%", "retrieval_channels": ["web"]}]
    state = run_agent(graph(Retriever(), LLM(route="websearch"), Search(wrong_bank)),
                      "工商银行今日利率？")
    assert state["status"] == "no_evidence"
    assert state["relevant_documents"] == []


def test_web_answer_mismatch_stops_after_one_search():
    class OffTopicLLM(LLM):
        def chat(self, messages, **kwargs):
            if "答案完整性校验器" in messages[0]["content"]:
                return '{"answers_question": false}'
            return super().chat(messages, **kwargs)

    search = Search(WEB)
    state = run_agent(graph(Retriever(), OffTopicLLM(route="websearch"), search),
                      "今日利率？")
    assert state["status"] == "no_evidence"
    assert search.queries == ["今日利率？"]


def test_extractive_web_fallback_shows_source_url():
    answer = AgentNodes._extractive_answer(WEB)
    assert "https://example.com/a" in answer
    assert "公开网页" in answer


def test_persisted_web_evidence_keeps_source_url():
    raw = {"document_version": "web-current", "block_id": "web:1",
           "source_doc": "公告", "source_url": "https://example.com/a",
           "content": "利率 3%", "retrieval_channels": ["web"]}
    evidence = evidence_from_v2(raw, "snapshot")
    assert evidence.metadata["source_url"] == "https://example.com/a"


def test_tavily_search_sends_only_query_and_filters_unsafe_sources():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size):
            return json.dumps({"results": [
                {"url": "http://127.0.0.1/private", "title": "internal", "content": "secret"},
                {"url": "https://example.com/report", "title": "报告", "content": "利率 3%"},
            ]}).encode()

    captured = {}

    def fake_open(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        return Response()

    with patch("src.system.web_search.urlopen", fake_open):
        results = TavilySearch("test-key").search("今日利率？")
    assert captured["body"]["query"] == "今日利率？"
    assert "secret" not in str(captured["body"])
    assert captured["authorization"] == "Bearer test-key"
    assert captured["timeout"] <= 10
    assert [item["source_url"] for item in results] == ["https://example.com/report"]


def test_tavily_domain_allowlist_rejects_secondary_and_lookalike_domains():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, _size):
            return json.dumps({"results": [
                {"url": "https://www.pbc.gov.cn/report", "content": "官方报告"},
                {"url": "https://pbc.gov.cn.evil.example/report", "content": "仿冒"},
                {"url": "https://news.example/report", "content": "转载"},
            ]}).encode()

    with patch("src.system.web_search.urlopen", return_value=Response()):
        result = TavilySearch("key", allowed_domains=("pbc.gov.cn",)).search("报告")
    assert [item["source_url"] for item in result] == ["https://www.pbc.gov.cn/report"]


def test_production_web_requires_explicit_domain_policy():
    with patch.dict("os.environ", {"APP_ENV": "production", "TAVILY_API_KEY": "key",
                                   "FINANCE_WEB_ALLOWED_DOMAINS": ""}):
        assert TavilySearch.from_env() is None
    with patch.dict("os.environ", {"APP_ENV": "production", "TAVILY_API_KEY": "key",
                                   "FINANCE_WEB_ALLOWED_DOMAINS": "pbc.gov.cn"}):
        assert TavilySearch.from_env().allowed_domains == ("pbc.gov.cn",)
