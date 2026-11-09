import unittest

from src.contracts import ContractError, ErrorCode, ModelSnapshot
from src.models.gateway import (
    ModelChunk,
    ModelGateway,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleTransport,
)


def snapshot():
    return ModelSnapshot(
        snapshot_id="sha256:test",
        profile_name="test",
        provider="fake",
        requested_model="fake-1",
        actual_model="fake-1",
    )


class FakeTransport:
    provider = "fake"

    def __init__(self, responses=None, stream_items=None):
        self.responses = list(responses or [])
        self.stream_items = list(stream_items or [])
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def stream(self, request):
        self.requests.append(request)
        for item in self.stream_items:
            if isinstance(item, BaseException):
                raise item
            yield item


class ModelGatewayTests(unittest.TestCase):
    def test_transport_keeps_truncation_and_known_usage(self):
        transport = OpenAICompatibleTransport(api_key="unused", base_url="https://example.test", provider="ark")
        transport._post = lambda *_args, **_kwargs: ({
            "model": "test-model", "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}, {})
        response = transport.complete(ModelRequest(messages=[], requested_model="test-model"))
        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.total_tokens, 30)

    def test_generate_records_model_and_latency(self):
        transport = FakeTransport(
            responses=[
                ModelResponse(
                    text="答案",
                    requested_model="fake-1",
                    actual_model="fake-1",
                    provider="fake",
                    request_id="r1",
                )
            ]
        )
        gateway = ModelGateway(snapshot(), transport, sleep=lambda _: None)

        response = gateway.generate(
            messages=[{"role": "user", "content": "问题"}],
            temperature=0.0,
            max_tokens=32,
        )

        self.assertEqual(response.text, "答案")
        self.assertEqual(response.provider, "fake")
        self.assertEqual(response.actual_model, "fake-1")
        self.assertGreaterEqual(response.latency_ms, 0)
        self.assertEqual(transport.requests[0].temperature, 0.0)

    def test_timeout_is_retried_once(self):
        transport = FakeTransport(
            responses=[
                TimeoutError("slow"),
                ModelResponse(
                    text="ok",
                    requested_model="fake-1",
                    actual_model="fake-1",
                    provider="fake",
                    request_id="r2",
                ),
            ]
        )
        gateway = ModelGateway(
            snapshot(), transport, max_retries=1, sleep=lambda _: None
        )

        self.assertEqual(gateway.generate(messages=[]).text, "ok")
        self.assertEqual(len(transport.requests), 2)

    def test_invalid_request_is_not_retried(self):
        transport = FakeTransport(
            responses=[ContractError(ErrorCode.INVALID_REQUEST, "bad")]
        )
        gateway = ModelGateway(
            snapshot(), transport, max_retries=2, sleep=lambda _: None
        )

        with self.assertRaises(ContractError) as raised:
            gateway.generate(messages=[])

        self.assertEqual(raised.exception.code, ErrorCode.INVALID_REQUEST)
        self.assertEqual(len(transport.requests), 1)

    def test_generate_json_accepts_fenced_object(self):
        transport = FakeTransport(
            responses=[
                ModelResponse(
                    text='```json\n{"answer": "ok"}\n```',
                    requested_model="fake-1",
                    actual_model="fake-1",
                    provider="fake",
                    request_id="r3",
                )
            ]
        )
        gateway = ModelGateway(snapshot(), transport, sleep=lambda _: None)

        self.assertEqual(gateway.generate_json(messages=[]), {"answer": "ok"})

    def test_stream_preserves_chunk_order(self):
        transport = FakeTransport(
            stream_items=[
                ModelChunk(text="答", index=0),
                ModelChunk(text="案", index=1),
            ]
        )
        gateway = ModelGateway(snapshot(), transport, sleep=lambda _: None)

        self.assertEqual(
            [chunk.text for chunk in gateway.stream(messages=[])],
            ["答", "案"],
        )

    def test_stream_interruption_has_stable_error_code(self):
        transport = FakeTransport(
            stream_items=[ModelChunk(text="答", index=0), RuntimeError("closed")]
        )
        gateway = ModelGateway(snapshot(), transport, sleep=lambda _: None)
        iterator = gateway.stream(messages=[])

        self.assertEqual(next(iterator).text, "答")
        with self.assertRaises(ContractError) as raised:
            next(iterator)
        self.assertEqual(raised.exception.code, ErrorCode.MODEL_UNAVAILABLE)

    def test_transport_repr_does_not_disclose_api_key(self):
        secret = "secret-key-for-test"
        transport = OpenAICompatibleTransport(
            api_key=secret,
            base_url="https://example.test",
            provider="deepseek",
        )

        self.assertNotIn(secret, repr(transport))


if __name__ == "__main__":
    unittest.main()
