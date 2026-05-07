"""
Token and cost tracking for LLM API calls

Used for RQ3 cost-effectiveness analysis.
Supports multiple LLM providers (DeepSeek, OpenAI, etc.)
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from agentic_szz.config.settings import get_model_pricing


@dataclass
class TokenUsage:
    """Token usage for a single API call"""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # Tokens served from cache
    model: str = "deepseek-chat"  # Model name for pricing lookup

    @classmethod
    def from_pydantic_ai(cls, usage, model: str = "deepseek-chat") -> "TokenUsage":
        """Create TokenUsage from pydantic-ai Usage object.

        pydantic-ai exposes cache_read_tokens which maps to
        DeepSeek's prompt_cache_hit_tokens (10x cheaper!).
        """
        if usage is None:
            return cls(model=model)
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cached_tokens=getattr(usage, "cache_read_tokens", 0) or 0,
            model=model,
        )

    @property
    def uncached_tokens(self) -> int:
        return self.input_tokens - self.cached_tokens

    @property
    def cost_usd(self) -> float:
        """Calculate cost using model-specific pricing."""
        pricing = get_model_pricing(self.model)
        return (
            self.uncached_tokens * pricing["input"]
            + self.cached_tokens * pricing["cached"]
            + self.output_tokens * pricing["output"]
        )


@dataclass
class CaseMetrics:
    """Metrics for a single test case"""

    case_id: str
    bfc: str
    ground_truth_bics: List[str]
    predicted_bics: List[str] = field(default_factory=list)
    success: bool = False

    # Resource usage
    steps: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    time_seconds: float = 0.0

    # Per-stage breakdown
    stage1_time: float = 0.0
    stage2_time: float = 0.0
    stage3_time: float = 0.0

    @property
    def cost_usd(self) -> float:
        return self.token_usage.cost_usd


class TokenTracker:
    """Track token usage across multiple API calls"""

    def __init__(self):
        self.total = TokenUsage()
        self.calls: List[TokenUsage] = []
        self._start_time: float = 0.0

    def start_timer(self):
        self._start_time = time.time()

    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    def add_usage(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0):
        """Record token usage from an API call"""
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )
        self.calls.append(usage)
        self.total.input_tokens += input_tokens
        self.total.output_tokens += output_tokens
        self.total.cached_tokens += cached_tokens

    def add_from_response(self, response):
        """Extract and record usage from OpenAI-style API response"""
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            self.add_usage(
                input_tokens=getattr(usage, "prompt_tokens", 0),
                output_tokens=getattr(usage, "completion_tokens", 0),
                cached_tokens=getattr(usage, "prompt_cache_hit_tokens", 0),
            )

    def reset(self):
        """Reset tracker for new case"""
        self.total = TokenUsage()
        self.calls = []
        self._start_time = 0.0

    @property
    def total_cost(self) -> float:
        return self.total.cost_usd

    @property
    def num_calls(self) -> int:
        return len(self.calls)
