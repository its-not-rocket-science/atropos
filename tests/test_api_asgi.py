from __future__ import annotations

from importlib.util import find_spec

import pytest

if find_spec("fastapi") is None:
    pytestmark = pytest.mark.skip(reason="fastapi is not installed")


def test_assert_required_health_routes_passes_for_runtime_app() -> None:
    from atroposlib.api.asgi import _assert_required_health_routes
    from atroposlib.api.server import build_runtime_app
    from atroposlib.api.storage import InMemoryStore

    app = build_runtime_app(store=InMemoryStore())

    _assert_required_health_routes(app)


def test_assert_required_health_routes_rejects_missing_readiness() -> None:
    from fastapi import FastAPI

    from atroposlib.api.asgi import _assert_required_health_routes

    app = FastAPI()
    app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    app.add_api_route("/health/live", lambda: {"status": "alive"}, methods=["GET"])
    app.add_api_route("/health/dependencies", lambda: {"status": "ok"}, methods=["GET"])

    with pytest.raises(ValueError, match="/health/ready"):
        _assert_required_health_routes(app)
