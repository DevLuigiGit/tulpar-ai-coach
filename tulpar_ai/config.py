"""Settings. Everything comes from env / `.env`; nothing is hard-coded to a production system.

Two modes:
  demo   — self-contained: demo trainer and clients from fixtures/, plan changes and the food diary live
           in this service's own SQLite. Runs anywhere without Tulpar. DEFAULT.
  tulpar — talks to a Tulpar instance over its HTTP API (+ read-only SQL). Meant for a LOCAL Tulpar
           stand only. It is never enabled implicitly, and it refuses a URL that looks like production.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"


def _export_env_file() -> None:
    """Make `.env` visible to libraries that read os.environ directly (LangSmith tracing)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_export_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    ai_data_dir: Path = ROOT / "data"

    # ── integration with Tulpar ────────────────────────────────────────────────
    tulpar_mode: str = "demo"  # demo | tulpar
    tulpar_api_url: str = ""  # local stand only, e.g. http://localhost:8088
    tulpar_jwt_secret: str = ""  # JWT_SECRET of THAT instance
    tulpar_database_url: str = ""  # read-only DSN of THAT instance (postgresql://...)

    # ── auth ───────────────────────────────────────────────────────────────────
    jwt_secret: str = "dev-only-tulpar-ai-coach-change-me-0123456789abcdef"
    internal_secret: str = "dev-only-internal-secret-change-me"
    allow_demo_login: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:8089"

    # ── LLM providers: a chain "provider:model,provider:model" per role ────────
    ollama_api_key: str = ""
    ollama_url: str = "https://ollama.com"
    groq_api_key: str = ""
    groq_url: str = "https://api.groq.com/openai/v1"
    route_models: str = "ollama:minimax-m3,groq:openai/gpt-oss-20b"
    text_models: str = "ollama:minimax-m3,groq:openai/gpt-oss-120b"
    vision_models: str = "ollama:kimi-k2.7-code,ollama:kimi-k2.6,ollama:minimax-m3"
    judge_models: str = "ollama:qwen3.5:397b,groq:openai/gpt-oss-120b"
    stt_model: str = "whisper-large-v3-turbo"
    llm_timeout_s: float = 60.0

    # ── voice replies (edge-tts, no key): bot answers a voice note with voice, web has a play button ──
    tts_enabled: bool = True
    tts_voice: str = "ru-RU-SvetlanaNeural"
    tts_voice_kk: str = "kk-KZ-AigulNeural"  # picked automatically for Kazakh text
    tts_rate: str = "+0%"
    tts_max_chars: int = 600
    tts_timeout_s: float = 30.0

    # ── hyperparameters (chosen by experiments, see EVALS.md) ──────────────────
    route_temperature: float = 0.0
    route_max_tokens: int = 200
    answer_temperature: float = 0.0
    answer_top_p: float = 0.9
    answer_max_tokens: int = 400
    draft_temperature: float = 0.2
    draft_top_p: float = 0.9
    draft_max_tokens: int = 1500
    vision_temperature: float = 0.0
    vision_max_tokens: int = 300

    # ── RAG ────────────────────────────────────────────────────────────────────
    jina_api_key: str = ""
    embed_model: str = "jina-embeddings-v3"
    rerank_model: str = "jina-reranker-v2-base-multilingual"
    rag_embedder: str = "auto"  # auto | jina | local
    rag_top_k: int = 20
    rag_top_n: int = 4
    rag_rerank: str = "auto"  # auto: on for the local embedder, off for Jina — both by A/B (EVALS.md); on | off
    rag_min_score: float = 0.2
    rag_max_rewrites: int = 2
    qdrant_url: str = ""  # empty: embedded Qdrant in AI_DATA_DIR/qdrant; http://qdrant:6333 or a Qdrant Cloud URL
    qdrant_api_key: str = ""

    # ── semantic answer cache (question branch only; thresholds by evals/cache_eval.py, EVALS.md) ───
    # The threshold alone cannot stop one-word flips (женщина/мужчина 0.989, в неделю/в день 0.983 with Jina); the
    # lexical guard in rag/cache_guard.py does. With the guard the lowest threshold with zero wrong hits is 0.91, but
    # a flip outside the guard's lists (узким/широким хватом) scores 0.909, so 0.95 keeps a margin above it.
    answer_cache: bool = True
    answer_cache_min_score: float = 0.95  # Jina embedder
    answer_cache_min_score_local: float = 0.99  # hashing embedder: lexical, so in practice only verbatim repeats
    answer_cache_ttl_h: float = 72.0

    # ── abuse protection (in-memory token buckets; one process = one set of buckets) ──
    rate_limit_enabled: bool = True
    chat_rate_per_min: float = 10.0  # sustained messages per minute per (user, IP)
    chat_burst: int = 10
    login_rate_per_min: float = 10.0  # demo logins per minute per IP
    login_burst: int = 20  # a classroom behind one NAT must still get in
    trainer_rate_per_min: float = 4.0  # program drafts and edits: several LLM calls each
    trainer_burst: int = 6
    trust_forwarded_for: bool = True  # behind Railway's proxy the client IP is in X-Forwarded-For

    # ── Telegram bot (this project's own bot, not Tulpar's) ───────────────────
    telegram_bot_token: str = ""
    trainer_telegram_ids: str = ""  # comma separated numeric Telegram ids of trainers
    webapp_url: str = ""  # public https URL of the web app for the Mini App; empty: https://$RAILWAY_PUBLIC_DOMAIN
    telegram_auth_max_age_s: int = 86400  # how old WebApp.initData may be at login

    # ── MCP client → service ───────────────────────────────────────────────────
    service_url: str = "http://localhost:8089"

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trainer_tg_ids(self) -> set[str]:
        return {x.strip() for x in self.trainer_telegram_ids.split(",") if x.strip()}

    @property
    def mini_app_url(self) -> str:
        """https URL for the bot's Mini App buttons, or "" (Telegram accepts only https)."""
        url = self.webapp_url.strip()
        if not url and os.environ.get("RAILWAY_PUBLIC_DOMAIN"):
            url = "https://" + os.environ["RAILWAY_PUBLIC_DOMAIN"].strip().strip("/")
        return url.rstrip("/") + "/" if url.startswith("https://") else ""

    @property
    def is_real_mode(self) -> bool:
        return self.tulpar_mode == "tulpar"

    def chain(self, role: str) -> list[tuple[str, str]]:
        raw = getattr(self, f"{role}_models")
        out = []
        for item in raw.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            provider, model = item.split(":", 1)
            out.append((provider.strip(), model.strip()))
        return out

    def provider_ready(self, provider: str) -> bool:
        return bool({"ollama": self.ollama_api_key, "groq": self.groq_api_key}.get(provider))

    def data_path(self, *parts: str) -> Path:
        p = Path(self.ai_data_dir).joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    Path(s.ai_data_dir).mkdir(parents=True, exist_ok=True)
    if s.is_real_mode:
        url = s.tulpar_api_url.lower()
        if not url or any(bad in url for bad in ("railway.app", "pages.dev", "tulpar-production")):
            raise RuntimeError(
                "TULPAR_MODE=tulpar is only allowed against a LOCAL Tulpar stand "
                f"(TULPAR_API_URL={s.tulpar_api_url!r}). Production is out of scope for this project."
            )
    on_server = bool(os.environ.get("RAILWAY_ENVIRONMENT")) or s.app_env == "production"
    if on_server and (s.jwt_secret.startswith("dev-only") or s.internal_secret.startswith("dev-only")):
        raise RuntimeError("Set JWT_SECRET and INTERNAL_SECRET on a server: the defaults are public in the repository.")
    return s
