"""Builds the API from the settings. Modal and local development both use this.

uv run uvicorn engine.main:build_app --factory --reload --port 8000
"""

from sqlalchemy import create_engine

from engine.agent import make_chat
from engine.api import create_app
from engine.config import get_settings


def build_app():
    cfg = get_settings()
    chat = make_chat(cfg.llm_base_url, cfg.models_api_key, cfg.llm_model, patience_s=40)  # a cold model: fail fast, the page retries
    return create_app(cfg, create_engine(cfg.database_url, pool_pre_ping=True), chat)
