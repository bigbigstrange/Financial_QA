import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import graph.graph as graph_module
from graph.consts import MAX_ITERATIONS
from graph.runtime import GraphTimeoutError, invoke_with_timeout


class Chain:
    """按顺序返回预设值；用完后重复最后一个，避免测试里出现 StopIteration。"""

    def __init__(self, values):
        self.values = list(values)

    def invoke(self, _payload):
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]


class RetryLimitTests(unittest.TestCase):
    def _run(self, *, generate_impl, grounded_values, useful_values):
        def retrieve(state):
            return {"question": state["question"], "documents": ["evidence"]}

        def grade(state):
            return {**state, "web_search": False}

        router = Chain([SimpleNamespace(datasource="vectorstore")])
        grounded = Chain([SimpleNamespace(binary_score=v) for v in grounded_values])
        useful = Chain([SimpleNamespace(binary_score=v) for v in useful_values])

        with patch.multiple(
            graph_module,
            retrieve=retrieve,
            grade_documents=grade,
            generate=generate_impl,
            question_router=router,
            hallucination_grader=grounded,
            answer_grader=useful,
        ):
            return graph_module.build_app().invoke(
                {"question": "agent memory"},
                {"recursion_limit": 10},
            )

    def test_gives_up_after_max_iterations(self):
        counter = {"generate": 0}

        def generate(state):
            counter["generate"] += 1
            return {
                **state,
                "generation": f"draft-{counter['generate']}",
                "iterations": counter["generate"],
            }

        result = self._run(
            generate_impl=generate,
            grounded_values=[False, False, False, False],
            useful_values=[True],
        )

        self.assertEqual(counter["generate"], MAX_ITERATIONS)
        self.assertTrue(result["refused"])
        self.assertIn("拒答", result["generation"])

    def test_retries_once_then_succeeds(self):
        counter = {"generate": 0}

        def generate(state):
            counter["generate"] += 1
            return {
                **state,
                "generation": f"draft-{counter['generate']}",
                "iterations": counter["generate"],
            }

        result = self._run(
            generate_impl=generate,
            grounded_values=[False, True],
            useful_values=[True],
        )

        self.assertEqual(counter["generate"], 2)
        self.assertEqual(result["generation"], "draft-2")
        self.assertFalse(result.get("refused", False))


class TimeoutTests(unittest.TestCase):
    class _SlowGraph:
        def invoke(self, *_args, **_kwargs):
            time.sleep(0.5)
            return {"generation": "too late"}

    def test_invoke_with_timeout_raises(self):
        with self.assertRaises(GraphTimeoutError):
            invoke_with_timeout(self._SlowGraph(), {"question": "q"}, timeout=0.05)

    def test_invoke_with_timeout_passes_through(self):
        class _FastGraph:
            def __init__(self):
                self.config = None

            def invoke(self, _inputs, config):
                self.config = config
                return {"generation": "ok"}

        graph = _FastGraph()
        result = invoke_with_timeout(graph, {"question": "q"}, timeout=5, recursion_limit=8)

        self.assertEqual(result["generation"], "ok")
        self.assertEqual(graph.config["recursion_limit"], 8)


if __name__ == "__main__":
    unittest.main()
