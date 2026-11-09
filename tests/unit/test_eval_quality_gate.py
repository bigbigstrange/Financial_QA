from unittest.mock import Mock, patch

import pytest

from app.services.eval_service import EvalService


def test_unreviewed_gold_blocks_eval_before_model_reload(tmp_path):
    review = tmp_path / "review.jsonl"
    review.write_text("pending", encoding="utf-8")
    service = EvalService()
    with (patch("app.services.eval_service.DEFAULT_REVIEW", review),
          patch("app.services.eval_service.validate", return_value={
              "quality_gate_ready": False, "pending_cases": 500, "errors": []}),
          patch("app.services.eval_service.qa_runtime.reload") as reload):
        with pytest.raises(ValueError, match="禁止付费批量评测"):
            service.run_sync(Mock())
    reload.assert_not_called()
