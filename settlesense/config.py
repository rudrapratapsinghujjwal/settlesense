"""
SettleSense — Configuration Management
Loads environment variables and provides typed config objects.
Never logs or prints secrets.
"""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env", override=False)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RazorpayConfig:
    key_id: str
    key_secret: str

    @property
    def is_configured(self) -> bool:
        return bool(self.key_id and self.key_secret
                    and not self.key_id.startswith("rzp_test_XXX"))

    @property
    def is_test_mode(self) -> bool:
        return self.key_id.startswith("rzp_test_")


@dataclass(frozen=True)
class LLMConfig:
    provider: str          # "anthropic" | "openai" | "mock"
    model: str
    anthropic_api_key: str
    openai_api_key: str

    @property
    def is_configured(self) -> bool:
        if self.provider == "mock":
            return True
        if self.provider == "anthropic":
            return bool(self.anthropic_api_key
                        and not self.anthropic_api_key.startswith("sk-ant-XXX"))
        if self.provider == "openai":
            return bool(self.openai_api_key
                        and not self.openai_api_key.startswith("sk-XXX"))
        return False


@dataclass(frozen=True)
class AppConfig:
    random_seed: int
    confidence_threshold: float
    log_level: str
    razorpay: RazorpayConfig
    llm: LLMConfig
    project_root: Path = field(default_factory=lambda: _project_root)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def db_path(self) -> Path:
        return self.project_root / "settlesense.db"


def load_config() -> AppConfig:
    """Load and validate configuration from environment variables."""
    razorpay_cfg = RazorpayConfig(
        key_id=os.getenv("RAZORPAY_KEY_ID", ""),
        key_secret=os.getenv("RAZORPAY_KEY_SECRET", ""),
    )

    # Auto-detect LLM provider if keys exist
    provider = os.getenv("LLM_PROVIDER", "mock")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    def _valid_anthropic_key(k: str) -> bool:
        # Anthropic keys: sk-ant-api03-... or sk-ant-...  (typically 90+ chars)
        return bool(k) and k.startswith("sk-ant-") and len(k) > 40

    def _valid_openai_key(k: str) -> bool:
        # OpenAI keys: sk-...  or sk-proj-...
        return bool(k) and k.startswith("sk-") and len(k) > 30

    if provider == "mock":
        if _valid_anthropic_key(anthropic_key):
            provider = "anthropic"
            logger.info("Auto-detected ANTHROPIC_API_KEY — using anthropic provider.")
        elif _valid_openai_key(openai_key):
            provider = "openai"
            logger.info("Auto-detected OPENAI_API_KEY — using openai provider.")
        else:
            if anthropic_key or openai_key:
                logger.warning(
                    "API key found but format is invalid (Anthropic keys start with 'sk-ant-', "
                    "OpenAI keys start with 'sk-'). Staying in mock mode."
                )
            else:
                logger.warning(
                    "No LLM API key found. LLM_PROVIDER=mock. "
                    "AI classification will use deterministic mock responses. "
                    "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable real AI."
                )

    llm_cfg = LLMConfig(
        provider=provider,
        model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
        anthropic_api_key=anthropic_key,
        openai_api_key=openai_key,
    )

    cfg = AppConfig(
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.70")),

        log_level=LOG_LEVEL,
        razorpay=razorpay_cfg,
        llm=llm_cfg,
    )

    # Log configuration status (no secrets)
    logger.info(
        "Config loaded | razorpay=%s | llm_provider=%s | llm_configured=%s | seed=%d",
        "configured" if cfg.razorpay.is_configured else "MISSING",
        cfg.llm.provider,
        cfg.llm.is_configured,
        cfg.random_seed,
    )

    return cfg


# Module-level singleton
config = load_config()
