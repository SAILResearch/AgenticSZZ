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
LLM_PRICING = {
    # DeepSeek-V3 (official API)
    "deepseek-chat": {
        "input": 2.8e-7,    # $0.28 / 1M tokens (cache miss)
        "cached": 2.8e-8,   # $0.028 / 1M tokens (cache hit)
        "output": 4.2e-7,   # $0.42 / 1M tokens
    },
    # GPT-4o-mini (OpenAI)
    "gpt-4o-mini": {
        "input": 1.5e-7,    # $0.15 / 1M tokens
        "cached": 7.5e-8,   # $0.075 / 1M tokens (cached)
        "output": 6.0e-7,   # $0.60 / 1M tokens
    },
}


def get_model_pricing(model: str) -> dict:
    """Get pricing for a model. Falls back to deepseek-chat if unknown."""
    # Normalize model name (handle prefixes like "openai:" or "deepseek:")
    model_name = model.split(":")[-1] if ":" in model else model
    return LLM_PRICING.get(model_name, LLM_PRICING["deepseek-chat"])


@dataclass
class LLMConfig:
    """LLM configuration - supports DeepSeek and OpenAI-compatible APIs
    
    Supports multiple providers:
    - DeepSeek: Set LLM_BASE_URL=https://api.deepseek.com, LLM_MODEL=deepseek-chat
    - OpenAI: Set LLM_BASE_URL=https://api.openai.com, LLM_MODEL=gpt-4o-mini
    
    Environment variables:
    - LLM_MODEL: Model name (default: deepseek-chat)
    - LLM_BASE_URL: API base URL (default: https://api.deepseek.com)
    - DEEPSEEK_API_KEY: DeepSeek API key (used when model contains 'deepseek')
    - OPENAI_API_KEY: OpenAI API key (used for gpt-* models)
    """

    # Model settings
    model: str = field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "deepseek-chat")
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
        
        # Select API key based on model type
        if self.api_key is None:
            if "deepseek" in self.model.lower():
                self.api_key = os.environ.get("DEEPSEEK_API_KEY")
            else:
                # For OpenAI models (gpt-4o, gpt-4o-mini, etc.)
                self.api_key = os.environ.get("OPENAI_API_KEY")

    # Generation parameters
    temperature: float = 0.0
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
