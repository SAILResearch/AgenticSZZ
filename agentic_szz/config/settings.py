"""
Settings and configuration for AgenticSZZ

Centralized configuration for:
- LLM (DeepSeek-V3.2 / OpenAI compatible)
- Neo4j (for Graphiti TKG)
- Pipeline stages

API keys are loaded from .env file (never committed to git).
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

# Load .env file for API keys
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Try loading from current directory
    load_dotenv()

# =============================================================================
# Path Constants
# =============================================================================
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
REPOS_DIR = os.path.join(PROJECT_ROOT, "dataset", "subject_repos")
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# LLM Pricing (per token, 2025-01)
# =============================================================================
# Model pricing lookup - add new models here
# Replication note: RQ1 was run with DeepSeek-V3.2 via the "deepseek-chat" alias
# (official API, before Apr 2026). RQ3 and later runs use explicit model names
# (deepseek-v4-flash / deepseek-v4-pro) per updated DeepSeek API (Apr 2026+).
# Historical entries below are kept so old result files remain interpretable.
# cost_usd is computed at run time and stored per case, so editing this table
# never changes results already on disk. Only deepseek-v4-pro is current; the
# rest are not called by the TOSEM revision runs.
LLM_PRICING = {
    # deepseek-chat: was DeepSeek-V3.2 (used for RQ1, run before Apr 2026).
    # Alias now points to deepseek-v4-flash non-thinking (deprecated 2026-07-24).
    # V3.2 pricing (pre-Apr 2026): "input": 2.8e-7, "cached": 2.8e-8, "output": 4.2e-7
    "deepseek-chat": {
        "input": 1.4e-7,    # $0.14 / 1M tokens (cache miss) — V4-Flash rates
        "cached": 2.8e-9,   # $0.0028 / 1M tokens (cache hit, reduced 10x on 2026-04-26)
        "output": 2.8e-7,   # $0.28 / 1M tokens
    },
    # GPT-4o-mini (OpenAI)
    "gpt-4o-mini": {
        "input": 1.5e-7,    # $0.15 / 1M tokens
        "cached": 7.5e-8,   # $0.075 / 1M tokens (cached)
        "output": 6.0e-7,   # $0.60 / 1M tokens
    },
    # DeepSeek-V4-Flash — explicit name (use instead of deprecated deepseek-chat alias)
    # Supports thinking mode (extra_body={"thinking": {"type": "enabled"}})
    "deepseek-v4-flash": {
        "input": 1.4e-7,    # $0.14 / 1M tokens (cache miss)
        "cached": 2.8e-9,   # $0.0028 / 1M tokens (cache hit, reduced 10x on 2026-04-26)
        "output": 2.8e-7,   # $0.28 / 1M tokens
    },
    # DeepSeek-V4-Pro: the primary model (D7), frontier model with thinking.
    #
    # OFF-PEAK RATES, confirmed against api-docs.deepseek.com/quick_start/pricing
    # on 2026-09-05 and cross-checked against the CNY table (Y0.15 / Y4.5 / Y13.5).
    # Peak is exactly double. Peak hours are 01:00 to 04:00 and 06:00 to 10:00
    # UTC, Monday to Friday only; every other hour, and all weekend, is off-peak.
    #
    # All evaluation runs are scheduled inside the off-peak window, so these are
    # the rates that apply. A run that spills into peak would be billed at double
    # and the recorded cost would understate it.
    #
    # The previous values here were the "75% discount until 2026-05-31" rates,
    # which expired before DeepSeek restructured to peak and off-peak billing on
    # 2026-08-16. They understated cache-hit cost by roughly 60x, which dominates
    # because this workload runs at about 77% cache hits.
    "deepseek-v4-pro": {
        "input": 6.6e-7,    # $0.66 / 1M tokens (cache miss, off-peak)
        "cached": 2.2e-8,   # $0.022 / 1M tokens (cache hit, off-peak)
        "output": 1.98e-6,  # $1.98 / 1M tokens (off-peak)
    },
    # DeepSeek-V4.1-Flash: the fifth RQ3 model (D9), served under the name
    # "deepseek-flash". The API lists only deepseek-flash and deepseek-v4-pro;
    # the retired deepseek-v4-flash alias is now routed to V4.1-Flash as well,
    # but its historical entry above is left as-is so old result files stay
    # interpretable. That entry is why this one is needed: get_model_pricing()
    # falls back to it, and its output rate is 2.1x too low for V4.1-Flash.
    #
    # OFF-PEAK RATES, confirmed against api-docs.deepseek.com/quick_start/pricing
    # on 2026-09-18 in the same fetch that reproduced the v4-pro rates above.
    # Peak is exactly double, in the same windows noted for v4-pro.
    "deepseek-flash": {
        "input": 1.5e-7,    # $0.15 / 1M tokens (cache miss, off-peak)
        "cached": 3.0e-9,   # $0.003 / 1M tokens (cache hit, off-peak)
        "output": 6.0e-7,   # $0.60 / 1M tokens (off-peak)
    },
    # RQ3 local models (served via vLLM) — no cost
    "qwen3-coder-next": {"input": 0.0, "cached": 0.0, "output": 0.0},
    "gemma-4-31b":      {"input": 0.0, "cached": 0.0, "output": 0.0},
    "devstral-small-2": {"input": 0.0, "cached": 0.0, "output": 0.0},
}


def get_model_pricing(model: str) -> dict:
    """Get pricing for a model. Falls back to deepseek-v4-flash if unknown."""
    # Normalize model name (handle prefixes like "openai:" or "deepseek:")
    model_name = model.split(":")[-1] if ":" in model else model
    return LLM_PRICING.get(model_name, LLM_PRICING["deepseek-v4-flash"])


@dataclass
class LLMConfig:
    """LLM configuration - supports DeepSeek and OpenAI-compatible APIs

    Replication note:
    - RQ1 used DeepSeek-V3.2 via model="deepseek-chat" (official API, before Apr 2026)
    - RQ3 and later use explicit model names: deepseek-v4-flash / deepseek-v4-pro

    Supports multiple providers:
    - DeepSeek: LLM_BASE_URL=https://api.deepseek.com, LLM_MODEL=deepseek-v4-flash
    - OpenAI: LLM_BASE_URL=https://api.openai.com, LLM_MODEL=gpt-4o-mini
    - Local vLLM: LLM_BASE_URL=http://host/v1, LLM_MODEL=<model-slug>

    Environment variables:
    - LLM_MODEL: Model name (default: deepseek-v4-flash)
    - LLM_BASE_URL: API base URL (default: https://api.deepseek.com)
    - DEEPSEEK_API_KEY: DeepSeek API key (used when model contains 'deepseek')
    - OPENAI_API_KEY: OpenAI API key (used for gpt-* models)
    """

    # Model settings
    model: str = field(
        # D7: DeepSeek retired V3.2, the paper's original primary model, so
        # deepseek-v4-pro is the primary for RQ1, RQ2 and the controlled
        # comparison. The old "deepseek-chat" alias is deprecated and no longer
        # denotes V3.2, so it must not be used as a default.
        default_factory=lambda: os.environ.get("LLM_MODEL", "deepseek-v4-pro")
    )
    api_key: Optional[str] = field(default=None)  # Set in __post_init__
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
    )

    def __post_init__(self):
        """Select API key and base_url based on model type."""
        # Auto-detect base_url from model if not explicitly set via env
        if not os.environ.get("LLM_BASE_URL"):
            if "gpt" in self.model.lower():
                self.base_url = "https://api.openai.com"
            elif "deepseek" in self.model.lower():
                self.base_url = "https://api.deepseek.com"

        # Select API key from environment based on provider
        if self.api_key is None:
            if "deepseek" in self.model.lower():
                self.api_key = os.environ.get("DEEPSEEK_API_KEY")
            elif any(k in self.model.lower() for k in ("gpt", "openai")):
                self.api_key = os.environ.get("OPENAI_API_KEY")
            else:
                # Local model (SAIL LLM cluster): personal key from .env, never committed
                self.api_key = os.environ.get("LOCAL_LLMS_API_KEY")

    # Generation parameters.
    # Read from LLM_TEMPERATURE so --temperature can reach worker processes,
    # the same way --model and --base-url already propagate.
    #
    # Default 1.0, matching DeepSeek's documented default, which is what the
    # original runs used before temperature was passed at all. Measured over
    # 100 cases per arm, temperature 0 makes the agent exhaust request_limit
    # far more often: 17% of cases return no prediction against 4% at 1.0.
    # On the full DS_LINUX set at temperature 0 the
    # rate reached 30%, burning $27 of $35 on cases that produced nothing.
    #
    # Section 5 claims temperature 0; that claim is corrected rather than
    # implemented, because implementing it measurably degrades the approach.
    # Sent explicitly rather than omitted so the setting is reproducible.
    temperature: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TEMPERATURE", "1.0"))
    )
    max_tokens: int = 4096
    
    @property
    def pricing(self) -> dict:
        """Get pricing for current model."""
        return get_model_pricing(self.model)


@dataclass
class Neo4jConfig:
    """Neo4j configuration for Graphiti TKG"""

    uri: str = field(
        default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    )
    user: str = field(default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j"))
    password: str = field(
        default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "password")
    )
    database: str = "neo4j"
    
    @classmethod
    def from_dict(cls, config: dict) -> "Neo4jConfig":
        """
        Create Neo4jConfig from worker database config dict.
        
        Used for per-worker database isolation in evaluation.
        
        Args:
            config: Dict with 'uri', 'user', 'password' keys
            
        Returns:
            Neo4jConfig instance with overridden values
        """
        return cls(
            uri=config.get("uri", "bolt://localhost:7687"),
            user=config.get("user", "neo4j"),
            password=config.get("password", "password"),
        )



@dataclass
class ResourceLimits:
    """Resource limits to prevent runaway costs"""

    max_steps: int = 50
    max_cost_usd: float = 1.0
    timeout_seconds: int = 3600


@dataclass
class TKGConfig:
    """TKG construction settings
    
    Design philosophy:
    - Use generous upper bounds for TKG construction to ensure coverage
    - Control agent visibility via candidate_limit (what list_candidates returns)
    - This avoids "magic threshold" formulas that are hard to justify
    """

    # Generous upper bounds for TKG construction (ensures high coverage)
    # These are intentionally high - the agent will only see top-K anyway
    ancestor_depth: int = 100  # Upper bound for ancestor traversal
    descendant_depth: int = 100  # Upper bound for descendant traversal

    # Hard limit on total candidates in TKG (performance constraint)
    max_candidates: int = 200
    
    # Agent visibility limit: how many candidates list_candidates() returns
    # Change this value for sensitivity analysis (e.g., 10, 20, 30)
    candidate_limit: int = 20
    
    # Context expansion for blame analysis.
    # Expand ±N lines around each deleted line to help find ancestor commits.
    # Default 0 (disabled); use --context-expand CLI flag to enable (e.g., 2).
    context_expand_lines: int = 0


@dataclass
class AgenticSZZConfig:
    """Main configuration - use get_config() to get singleton"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    tkg: TKGConfig = field(default_factory=TKGConfig)
    limits: ResourceLimits = field(default_factory=ResourceLimits)


# Singleton config
_config: Optional[AgenticSZZConfig] = None


def get_config(refresh: bool = False) -> AgenticSZZConfig:
    """Get global config singleton.
    
    Args:
        refresh: If True, recreate config from current env vars.
                 Use after changing LLM_MODEL or other env vars.
    """
    global _config
    if _config is None or refresh:
        _config = AgenticSZZConfig()
    return _config


def set_config(config: AgenticSZZConfig):
    """Set global config"""
    global _config
    _config = config
