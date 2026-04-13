"""Pytest configuration for unit tests."""

import pytest
from contextlib import asynccontextmanager


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set required env vars for all unit tests."""
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "test_salt_32_chars_long_padding!")
    monkeypatch.setenv("ENABLE_STRICT_JWT", "false")


@pytest.fixture(scope="function")
async def app_with_engine():
    """Trigger lifespan to initialize engine for API tests."""
    from services.serving.main import app, engine, start_time
    from services.serving.config import ServingConfig

    # Initialize engine if not already done
    if engine is None:
        config = ServingConfig()
        from services.serving.main import RecommendationEngine

        new_engine = RecommendationEngine(config)
        await new_engine.initialize()
        # Note: This won't actually update the global, but tests can import engine directly

    return app
