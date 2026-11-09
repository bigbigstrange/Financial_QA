"""One bounded real Tavily -> Ark graph acceptance probe (explicit --send)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.agent import build_graph, run_agent
from src.agent.nodes import AgentConfig
from src.system.llm_api import DeepSeekAPIClient
from src.system.web_search import TavilySearch


ROOT = Path(__file__).resolve().parents[1]


class _NoLocalRetriever:
    def search(self, *_args, **_kwargs):
        raise AssertionError("Web 分支不应访问本地检索器")


class _WebRouteArk:
    """Force only the routing decision; all later paid nodes use real Ark."""

    provider = "ark"
    enabled = True

    def __init__(self, client: DeepSeekAPIClient):
        self.client = client

    def chat(self, messages, *, node="chat", **kwargs):
        if node == "route_query":
            return json.dumps({"route": "websearch", "reason": "有界 Web 分支验收"},
                              ensure_ascii=False)
        return self.client.chat(messages, node=node, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--send", action="store_true", help="明确允许一次真实搜索与模型问答")
    parser.add_argument("--natural-route", action="store_true",
                        help="路由节点也由真实模型决定；可能不走 Web")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "output" / "acceptance" / "web_graph_pilot.json")
    args = parser.parse_args()
    if not args.send:
        raise SystemExit("预检模式：如需真实调用，显式传入 --send")
    load_dotenv(ROOT / ".env")
    web = TavilySearch.from_env()
    client = DeepSeekAPIClient()
    if web is None or not client.enabled or client.provider != "ark":
        raise SystemExit("真实 Tavily 或 Ark 未配置；未发起请求")

    question = "根据中国人民银行官网，2026年第一季度中国货币政策执行报告何时发布？"
    llm = client if args.natural_route else _WebRouteArk(client)
    graph = build_graph(_NoLocalRetriever(), llm, web_search=web,
                        config=AgentConfig(max_retry_retrieve=0,
                                           max_retry_generate=0))
    state = run_agent(graph, question, run_id=("acceptance-web-natural-20260924"
                                                if args.natural_route else
                                                "acceptance-web-graph-20260924"))
    sources = [{"title": item.get("source_doc"), "url": item.get("source_url")}
               for item in state.get("relevant_documents") or []]
    report = {
        "question": question,
        "route": state.get("route"),
        "route_forced": not args.natural_route,
        "status": state.get("status"),
        "verified_by_model": bool(state.get("verified")),
        "human_fact_checked": False,
        "trace": [{"node": item.get("node"), "decision": item.get("decision")}
                  for item in state.get("trace") or []],
        "sources": sources,
        "answer": state.get("generation"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "answer"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
