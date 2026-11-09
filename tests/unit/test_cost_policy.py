import os
import unittest
from unittest.mock import patch
from src.system.llm_api import DeepSeekAPIClient
from src.system.lightrag_client import LightRAGClient, LightRAGError
from src.system.lightrag_channel import build_lightrag_retriever


class CostPolicyTests(unittest.TestCase):
    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "1"})
    def test_llm_blocked_before_network(self):
        client = DeepSeekAPIClient(api_key="test")
        self.assertFalse(client.enabled)
        with patch.object(client.gateway, "generate") as call:
            with self.assertRaisesRegex(RuntimeError, "暂停"):
                client.chat([])
            call.assert_not_called()

    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "1"})
    def test_graph_query_and_ingest_blocked_before_network(self):
        client = LightRAGClient()
        self.assertIsNone(build_lightrag_retriever([], enabled=True))
        with patch("requests.request") as call:
            for path in ("/query/data", "/documents/text"):
                with self.assertRaisesRegex(LightRAGError, "暂停"):
                    client._request("POST", path, {})
            call.assert_not_called()

    @patch.dict(os.environ, {"FINANCE_DISABLE_PAID_API": "0"})
    def test_explicit_resume_enables_client(self):
        self.assertTrue(DeepSeekAPIClient(api_key="test").enabled)
