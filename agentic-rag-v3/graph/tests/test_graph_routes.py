import unittest
from types import SimpleNamespace
from unittest.mock import patch

import graph.graph as graph_module
from graph.nodes.web_search import ExternalSearchDisabled, web_search


class Chain:
    def __init__(self, values):
        self.values = iter(values)

    def invoke(self, _payload):
        return next(self.values)


class AgentGraphRouteTests(unittest.TestCase):
    def _build(self, *, router_values, grounded_values, useful_values, counter):
        def retrieve(state):
            return {"question": state["question"], "documents": ["evidence"]}

        def grade(state):
            return {**state, "web_search": False}

        def generate(state):
            counter["generate"] += 1
            return {**state, "generation": f"draft-{counter['generate']}"}

        router = Chain([SimpleNamespace(datasource=value) for value in router_values])
        grounded = Chain([SimpleNamespace(binary_score=value) for value in grounded_values])
        useful = Chain([SimpleNamespace(binary_score=value) for value in useful_values])

        patches = patch.multiple(
            graph_module,
            retrieve=retrieve,
            grade_documents=grade,
            generate=generate,
            question_router=router,
            hallucination_grader=grounded,
            answer_grader=useful,
        )
        return patches

    def test_vectorstore_useful_path_generates_once(self):
        counter = {"generate": 0}
        with self._build(
            router_values=["vectorstore"],
            grounded_values=[True],
            useful_values=[True],
            counter=counter,
        ):
            result = graph_module.build_app().invoke({"question": "agent memory"})
        self.assertEqual(counter["generate"], 1)
        self.assertEqual(result["generation"], "draft-1")

    def test_unsupported_draft_retries_before_ending(self):
        counter = {"generate": 0}
        with self._build(
            router_values=["vectorstore"],
            grounded_values=[False, True],
            useful_values=[True],
            counter=counter,
        ):
            result = graph_module.build_app().invoke(
                {"question": "agent memory"},
                {"recursion_limit": 8},
            )
        self.assertEqual(counter["generate"], 2)
        self.assertEqual(result["generation"], "draft-2")

    def test_external_search_fails_closed(self):
        with self.assertRaisesRegex(ExternalSearchDisabled, "disabled"):
            web_search({"question": "today's market", "documents": []})


if __name__ == "__main__":
    unittest.main()
