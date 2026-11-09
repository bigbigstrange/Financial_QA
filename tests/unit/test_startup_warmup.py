from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("manifest", [None, "C:/data/active/manifest.json"])
def test_main_warms_active_snapshot_without_first_request_reload(monkeypatch, manifest):
    main = import_module("app.main")
    uvicorn = import_module("uvicorn")
    calls = []
    monkeypatch.delenv("FINANCE_SKIP_WARMUP", raising=False)
    monkeypatch.setattr(main, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(main, "SessionLocal", lambda: nullcontext(object()))
    monkeypatch.setattr(main.run_service, "_knowledge_ref",
                        lambda db: SimpleNamespace(manifest_path=manifest) if manifest else None)
    monkeypatch.setattr(main.qa_runtime, "ensure_loaded",
                        lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append("serve"))

    main.main()

    assert calls == ["init_db", {"manifest_path": manifest}, "serve"]
