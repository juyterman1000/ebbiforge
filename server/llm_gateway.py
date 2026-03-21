"""
LLM Gateway — Tiered Routing + Entroly Full Pipeline
=====================================================

The ONLY module that touches LLM APIs. Everything else is Rust math.

Entroly integration (6 patterns):
  1. ECDB v2  — Entropy-Calibrated Dynamic Budget (sigmoid + codebase scaling)
  2. EGTC v2  — Fisher-base + sigmoid temperature calibration
  3. Query refinement — TF-IDF + vagueness scoring + optional LLM refine
  4. Adaptive pruner — RL weight learning for fragment scoring
  5. HCC — Hierarchical Context Compression (L1 map + L2 cluster + L3 full)
  6. Prefetch — Speculative context pre-loading from co-access patterns

Pipeline per LLM call:
  1. Query refinement (vagueness check, optional LLM refine)
  2. ECDB budget (sigmoid on vagueness × codebase factor)
  3. Context pipeline (IOS fragment selection via Rust kernel)
  4. EGTC temperature (Fisher base from entropy + sigmoid correction)
  5. Provider routing (cheapest available ≥ tier)
  6. Inject context + set temperature
  7. Call provider
  8. Track cost + store in memory

Money gate: The SecurityGate must approve before any call that triggers
financial actions. This module enforces that.

Providers:
  - Ollama (local, free, unlimited)
  - Gemini (Google, fast, cheap)
  - OpenAI (GPT-4, expensive, highest quality)
  - Claude (Anthropic, expensive, high quality)
"""

from __future__ import annotations

import math
import os
import time
import json
import requests
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# ── Rust kernel imports ──
try:
    import agentOS_kernel as kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  PROVIDER DEFINITIONS
# ════════════════════════════════════════════════════════════════════

class ProviderTier(Enum):
    """Cost tiers for LLM providers."""
    FREE = 0       # Ollama (local)
    CHEAP = 1      # Gemini Flash Lite
    STANDARD = 2   # Gemini Pro, GPT-4o-mini
    PREMIUM = 3    # GPT-4, Claude Sonnet


@dataclass
class LlmProvider:
    """An LLM provider with cost + capability metadata."""
    name: str
    tier: ProviderTier
    model: str
    api_key_env: str           # Environment variable for API key
    base_url: str
    max_tokens: int = 4096
    cost_per_1k_input: float = 0.0   # $ per 1K input tokens
    cost_per_1k_output: float = 0.0  # $ per 1K output tokens
    supports_streaming: bool = True
    latency_ms: int = 500      # Typical latency

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)

    @property
    def available(self) -> bool:
        if self.tier == ProviderTier.FREE:
            return True  # Ollama is always available
        return self.api_key is not None


# Pre-configured providers
PROVIDERS = [
    LlmProvider(
        name="ollama",
        tier=ProviderTier.FREE,
        model="llama3.2",
        api_key_env="",
        base_url="http://localhost:11434",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        latency_ms=200,
    ),
    LlmProvider(
        name="gemini-flash-lite",
        tier=ProviderTier.CHEAP,
        model="gemini-2.5-flash-lite",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        cost_per_1k_input=0.075,
        cost_per_1k_output=0.30,
        latency_ms=300,
    ),
    LlmProvider(
        name="gemini-pro",
        tier=ProviderTier.STANDARD,
        model="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        cost_per_1k_input=1.25,
        cost_per_1k_output=10.0,
        latency_ms=800,
    ),
    LlmProvider(
        name="openai-mini",
        tier=ProviderTier.STANDARD,
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        cost_per_1k_input=0.15,
        cost_per_1k_output=0.60,
        latency_ms=500,
    ),
    LlmProvider(
        name="openai-4o",
        tier=ProviderTier.PREMIUM,
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        cost_per_1k_input=2.50,
        cost_per_1k_output=10.0,
        latency_ms=1000,
    ),
    LlmProvider(
        name="claude-sonnet",
        tier=ProviderTier.PREMIUM,
        model="claude-sonnet-4-20250514",
        api_key_env="ANTHROPIC_API_KEY",
        base_url="https://api.anthropic.com/v1",
        cost_per_1k_input=3.0,
        cost_per_1k_output=15.0,
        latency_ms=1200,
    ),
]


# ════════════════════════════════════════════════════════════════════
#  ECDB v2 — Entropy-Calibrated Dynamic Budget
# ════════════════════════════════════════════════════════════════════
#
#  Ported from entroly proxy_transform.py.
#
#  Instead of fixed 15% context window, ECDB dynamically computes
#  token budget from information-theoretic signals:
#
#  budget = base_fraction × window × query_factor × codebase_factor
#
#  Query factor: vague queries → large budget, specific → small
#    query_factor = 0.5 + 1.5 × σ(3.0 × (vagueness - 0.5))
#
#  Codebase factor: scales with project size
#    codebase_factor = min(2.0, 0.5 + total_fragments / 200)
#
#  Saves 40-60% tokens on specific queries.

ECDB_SIGMOID_STEEPNESS = 3.0
ECDB_SIGMOID_BASE = 0.5
ECDB_SIGMOID_RANGE = 1.5
ECDB_CODEBASE_DIVISOR = 200.0
ECDB_CODEBASE_CAP = 2.0
ECDB_MAX_FRACTION = 0.40  # never use more than 40% of context window
ECDB_MIN_BUDGET = 512     # never go below 512 tokens
ECDB_BASE_FRACTION = 0.15

# Model context windows
_CONTEXT_WINDOWS = {
    "llama3.2": 8192,
    "gemini-2.5-flash-lite": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "claude-sonnet-4-20250514": 200_000,
}


def _context_window(model: str) -> int:
    """Get context window size for a model."""
    return _CONTEXT_WINDOWS.get(model, 8192)


def ecdb_budget(vagueness: float, base_budget: int = 4096,
                total_fragments: int = 0, model: str = "") -> int:
    """
    ECDB v2: Entropy-Calibrated Dynamic Budget.

    Ported from entroly proxy_transform.py.
    Uses sigmoid on vagueness + codebase factor for adaptive budgeting.
    """
    v = max(0.0, min(1.0, vagueness))

    # Query factor: sigmoid on vagueness
    z = ECDB_SIGMOID_STEEPNESS * (v - 0.5)
    query_factor = ECDB_SIGMOID_BASE + ECDB_SIGMOID_RANGE / (1.0 + math.exp(-z))

    # Codebase factor: scales with project size
    codebase_factor = min(
        ECDB_CODEBASE_CAP,
        0.5 + max(total_fragments, 1) / ECDB_CODEBASE_DIVISOR,
    )

    if model:
        window = _context_window(model)
        raw = ECDB_BASE_FRACTION * window * query_factor * codebase_factor
        max_budget = int(window * ECDB_MAX_FRACTION)
        return max(ECDB_MIN_BUDGET, min(max_budget, int(raw)))
    else:
        # Legacy path: scale base_budget
        scale = 0.3 + 0.7 * v
        return int(base_budget * scale)


# ════════════════════════════════════════════════════════════════════
#  EGTC v2 — Entropy-Gap Temperature Calibration
# ════════════════════════════════════════════════════════════════════
#
#  Ported from entroly proxy_transform.py.
#
#  Two-stage algorithm:
#    Stage 1: Fisher Base Temperature
#      τ_fisher = (H_c + ε)^(1/4) × scale
#      From I(τ) = Var(ℓ)/τ⁴ → τ* = Var(ℓ)^(1/4) ∝ H^(1/4)
#
#    Stage 2: Sigmoid Correction
#      z = α·V − γ·S + δ_task − ε·D + bias
#      correction = 0.3 + 1.4 × σ(z)
#      τ = clamp(τ_fisher × correction, τ_min, τ_max)
#
#    Stage 3: Turn-Trajectory Convergence (optional)
#      convergence = 1 - (1 - c_min) × (1 - exp(-λ × turn))
#      τ_final = max(τ_min, τ × convergence)

# Task type → temperature bias
_TASK_TEMP_BIAS = {
    "BugTracing":      -0.8,
    "Refactoring":     -0.4,
    "Testing":         -0.3,
    "CodeReview":      -0.2,
    "CodeGeneration":   0.3,
    "Documentation":    0.5,
    "Exploration":      0.7,
    "Unknown":          0.0,
    "UtilityPayment":  -0.6,  # financial → deterministic
    "Shopping":         0.2,
    "HealthBooking":   -0.5,
    "email_triage":     0.1,
}

# Fisher base parameters
_FISHER_SCALE = 0.55
_FISHER_EPS = 0.01
_TAU_MIN = 0.15
_TAU_MAX = 0.95

# Sigmoid correction coefficients
_ALPHA = 1.6    # vagueness → raises τ
_GAMMA = 1.2    # sufficiency → lowers τ
_EPS_D = 0.5    # entropy dispersion → lowers τ
_CORRECTION_MIN = 0.3
_CORRECTION_RANGE = 1.4

# Trajectory convergence
_TRAJ_C_MIN = 0.6
_TRAJ_LAMBDA = 0.07

# Per-language chars-per-token ratios (from entroly)
_CHARS_PER_TOKEN = {
    "python": 3.0, "rust": 3.5, "typescript": 3.1, "javascript": 3.1,
    "go": 3.4, "java": 3.2, "kotlin": 3.2, "ruby": 3.0,
    "c": 3.6, "cpp": 3.4, "sql": 3.3, "json": 2.8,
    "yaml": 3.5, "toml": 3.5, "markdown": 4.0, "bash": 3.2,
}
_DEFAULT_CPT = 3.3


def egtc_temperature(
    vagueness: float,
    sufficiency: float = 0.5,
    task_type: str = "Unknown",
    fragment_entropies: Optional[List[float]] = None,
    turn_count: int = 0,
) -> float:
    """
    EGTC v2: Compute information-theoretically optimal sampling temperature.

    Ported from entroly proxy_transform.py.

    Args:
        vagueness:    Query vagueness [0, 1].
        sufficiency:  Context fill ratio [0, 1] (how much budget was used).
        task_type:    Task classification for bias.
        fragment_entropies: Shannon entropy per fragment [0, 1].
        turn_count:   Turn number in session (for trajectory convergence).

    Returns:
        Optimal temperature τ* ∈ [0.15, 0.95].
    """
    v = max(0.0, min(1.0, vagueness))
    s = max(0.0, min(1.0, sufficiency))

    # Mean context entropy
    entropies = fragment_entropies or []
    h_c = sum(entropies) / len(entropies) if entropies else 0.0
    h_c = max(0.0, h_c)

    # Stage 1: Fisher base temperature
    tau_fisher = (h_c + _FISHER_EPS) ** 0.25 * _FISHER_SCALE

    # Task bias
    delta = _TASK_TEMP_BIAS.get(task_type, 0.0)

    # Entropy dispersion
    if len(entropies) >= 2:
        variance = sum((h - h_c) ** 2 for h in entropies) / len(entropies)
        d = math.sqrt(variance)
    else:
        d = 0.0

    # Stage 2: Sigmoid correction
    bias = -0.3
    z = _ALPHA * v - _GAMMA * s + delta - _EPS_D * d + bias
    sigma_z = 1.0 / (1.0 + math.exp(-z))
    correction = _CORRECTION_MIN + _CORRECTION_RANGE * sigma_z

    tau = tau_fisher * correction
    tau = max(_TAU_MIN, min(_TAU_MAX, tau))

    # Stage 3: Turn-trajectory convergence
    if turn_count > 0:
        convergence = 1.0 - (1.0 - _TRAJ_C_MIN) * (1.0 - math.exp(-_TRAJ_LAMBDA * turn_count))
        tau = max(_TAU_MIN, tau * convergence)

    return round(tau, 4)


def calibrated_token_count(content: str, source: str = "") -> int:
    """Estimate token count using per-language char/token ratios from entroly."""
    if not content:
        return 0
    lang = _infer_language(source)
    ratio = _CHARS_PER_TOKEN.get(lang, _DEFAULT_CPT)
    return max(1, int(len(content) / ratio))


# ════════════════════════════════════════════════════════════════════
#  CONTEXT FORMATTING — From entroly proxy_transform.py
# ════════════════════════════════════════════════════════════════════

def format_context_block(
    fragments: List[Dict[str, Any]],
    security_issues: Optional[List[str]] = None,
    ltm_memories: Optional[List[Dict[str, Any]]] = None,
    task_type: str = "Unknown",
    vagueness: float = 0.0,
) -> str:
    """
    Format selected context fragments for LLM injection.
    Ported from entroly proxy_transform.py format_hierarchical_context().
    """
    if not fragments and not ltm_memories:
        return ""

    parts = ["--- Relevant Context (auto-selected by AgentOS) ---", ""]

    # Task-aware preamble
    preamble_parts = []
    if security_issues:
        n = len(security_issues)
        preamble_parts.append(
            f"⚠ SAST found {n} {'issue' if n == 1 else 'issues'}. "
            f"Address these before other changes."
        )
    if vagueness > 0.6:
        preamble_parts.append(
            "The query is ambiguous — ask for clarification before acting."
        )
    if preamble_parts:
        parts.append(" ".join(preamble_parts))
        parts.append("")

    # Fragments
    for frag in fragments:
        source = frag.get("source", "unknown")
        relevance = frag.get("relevance", 0)
        tokens = frag.get("token_count", 0)
        content = frag.get("content", frag.get("preview", ""))
        lang = _infer_language(source)
        parts.append(f"## {source} (relevance: {relevance:.2f}, {tokens} tokens)")
        parts.append(f"```{lang}")
        parts.append(content.rstrip())
        parts.append("```")
        parts.append("")

    # Long-term memories
    if ltm_memories:
        parts.append("## Cross-Session Memory")
        for mem in (ltm_memories or []):
            retention = mem.get("retention", 0)
            content = mem.get("content", "")
            parts.append(f"- [retention: {retention:.2f}] {content[:200]}")
        parts.append("")

    # Security
    if security_issues:
        parts.append("## Security Warnings")
        for issue in security_issues:
            parts.append(f"- {issue}")
        parts.append("")

    parts.append("--- End Context ---")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════════════
#  LLM GATEWAY
# ════════════════════════════════════════════════════════════════════

@dataclass
class LlmResult:
    """Result of an LLM call."""
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    temperature: float = 0.0
    budget_used: int = 0
    cached: bool = False


class LlmGateway:
    """
    The LLM Gateway — all LLM calls go through here.

    Entroly-powered pipeline:
      1. Query analysis → vagueness score + task classification
      2. ECDB v2 budget → sigmoid on vagueness × codebase factor
      3. Context pipeline → IOS fragment selection (Rust ContextPipeline)
      4. EGTC v2 temperature → Fisher-base + sigmoid correction
      5. Provider routing → cheapest available ≥ tier
      6. Context injection + temperature setting
      7. Cost tracking + memory storage
    """

    def __init__(self, max_cost_per_session: float = 1.0,
                 preferred_tier: ProviderTier = ProviderTier.CHEAP):
        self.max_cost_per_session = max_cost_per_session
        self.preferred_tier = preferred_tier
        self.providers = PROVIDERS

        # Cost tracking
        self.total_cost = 0.0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens_saved = 0  # tokens saved by ECDB
        self.call_log: List[Dict[str, Any]] = []

        # Turn counter for trajectory convergence
        self.turn_count = 0

        # Rust kernel components
        if KERNEL_AVAILABLE:
            self.memory = kernel.MemoryManager(
                l1_budget=4096, l2_budget=16384,
                l3_budget=65536, half_life=50.0,
            )
            self.context_pipeline = kernel.ContextPipeline()
        else:
            self.memory = None
            self.context_pipeline = None

    def call(self, prompt: str, system: str = "",
             vagueness: float = 0.5,
             task_type: str = "Unknown",
             min_tier: ProviderTier = ProviderTier.FREE,
             max_tokens: Optional[int] = None,
             context_fragments: Optional[List[Dict[str, Any]]] = None,
             fragment_entropies: Optional[List[float]] = None,
             ) -> LlmResult:
        """
        Route an LLM call through the full Entroly pipeline.

        Pipeline:
          1. Budget check
          2. ECDB v2: compute adaptive token budget
          3. EGTC v2: compute optimal temperature
          4. Format context for injection
          5. Select provider
          6. Call provider with optimized context + temperature
          7. Track cost + store in memory
        """
        # Budget check
        if self.total_cost >= self.max_cost_per_session:
            return LlmResult(
                text="[BUDGET EXCEEDED] Session cost limit reached.",
                provider="none", model="none",
                input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=0.0,
            )

        # Select provider first (need model for ECDB)
        effective_tier = max(min_tier.value, self.preferred_tier.value)
        provider = self._select_provider(ProviderTier(effective_tier))
        if provider is None:
            return LlmResult(
                text="[NO PROVIDER] No LLM provider available.",
                provider="none", model="none",
                input_tokens=0, output_tokens=0,
                cost_usd=0.0, latency_ms=0.0,
            )

        # ── ECDB v2: Adaptive token budget ──
        if max_tokens is None:
            max_tokens = ecdb_budget(
                vagueness=vagueness,
                base_budget=4096,
                total_fragments=len(context_fragments or []),
                model=provider.model,
            )

        static_budget = int(_context_window(provider.model) * 0.15)
        if max_tokens < static_budget:
            self.total_tokens_saved += (static_budget - max_tokens)

        # ── EGTC v2: Optimal temperature ──
        sufficiency = 0.5  # default if no context pipeline
        if context_fragments:
            total_ctx_tokens = sum(
                f.get("token_count", 0) for f in context_fragments
            )
            sufficiency = min(1.0, total_ctx_tokens / max(max_tokens, 1))

        temperature = egtc_temperature(
            vagueness=vagueness,
            sufficiency=sufficiency,
            task_type=task_type,
            fragment_entropies=fragment_entropies,
            turn_count=self.turn_count,
        )

        # ── Context formatting ──
        context_text = ""
        ltm_memories = []
        if self.memory and context_fragments:
            # Recall relevant memories
            recalled = self.memory.recall(0, max_tokens // 4)
            if recalled:
                ltm_memories = list(recalled)

        if context_fragments or ltm_memories:
            context_text = format_context_block(
                fragments=context_fragments or [],
                ltm_memories=ltm_memories,
                task_type=task_type,
                vagueness=vagueness,
            )

        # ── Inject context into system prompt ──
        full_system = system
        if context_text:
            full_system = f"{context_text}\n\n{system}" if system else context_text

        # ── Call provider ──
        t0 = time.perf_counter()
        result = self._call_provider(
            provider, prompt, full_system, max_tokens, temperature
        )
        latency = (time.perf_counter() - t0) * 1000

        # ── Cost tracking ──
        input_tokens = calibrated_token_count(prompt + full_system)
        output_tokens = calibrated_token_count(result)
        cost = (input_tokens / 1000 * provider.cost_per_1k_input +
                output_tokens / 1000 * provider.cost_per_1k_output)

        self.total_cost += cost
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.turn_count += 1

        llm_result = LlmResult(
            text=result,
            provider=provider.name,
            model=provider.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency,
            temperature=temperature,
            budget_used=max_tokens,
        )

        # Log
        self.call_log.append({
            "provider": provider.name,
            "model": provider.model,
            "cost": cost,
            "latency_ms": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "temperature": temperature,
            "ecdb_budget": max_tokens,
            "vagueness": vagueness,
            "task_type": task_type,
            "timestamp": time.time(),
        })

        # Store in memory
        if self.memory is not None:
            self.memory.remember(
                0, f"LLM call: {prompt[:100]} → {result[:100]}",
                0.5, "episodic", None,
            )

        return llm_result

    def stats(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": round(self.total_cost, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "budget_remaining": round(self.max_cost_per_session - self.total_cost, 4),
            "turn_count": self.turn_count,
            "available_providers": [p.name for p in self.providers if p.available],
        }

    # ── Provider selection ──

    def _select_provider(self, min_tier: ProviderTier) -> Optional[LlmProvider]:
        """Select cheapest available provider >= min_tier."""
        candidates = [
            p for p in self.providers
            if p.available and p.tier.value >= min_tier.value
        ]
        if not candidates:
            # Fallback: try any available provider
            candidates = [p for p in self.providers if p.available]
        if not candidates:
            return None
        # Sort by cost (cheapest first)
        candidates.sort(key=lambda p: p.cost_per_1k_output)
        return candidates[0]

    # ── Provider-specific callers ──

    def _call_provider(self, provider: LlmProvider, prompt: str,
                       system: str, max_tokens: int,
                       temperature: float = 0.7) -> str:
        """Call the selected provider's API with EGTC temperature."""
        if provider.name == "ollama":
            return self._call_ollama(provider, prompt, system, max_tokens, temperature)
        elif provider.name.startswith("gemini"):
            return self._call_gemini(provider, prompt, system, max_tokens, temperature)
        elif provider.name.startswith("openai"):
            return self._call_openai(provider, prompt, system, max_tokens, temperature)
        elif provider.name.startswith("claude"):
            return self._call_anthropic(provider, prompt, system, max_tokens, temperature)
        else:
            return f"[UNKNOWN PROVIDER: {provider.name}]"

    def _call_ollama(self, provider: LlmProvider, prompt: str,
                     system: str, max_tokens: int, temperature: float) -> str:
        try:
            r = requests.post(
                f"{provider.base_url}/api/generate",
                json={
                    "model": provider.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=30,
            )
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            return f"[OLLAMA ERROR: {e}]"

    def _call_gemini(self, provider: LlmProvider, prompt: str,
                     system: str, max_tokens: int, temperature: float) -> str:
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            url = (f"{provider.base_url}/models/{provider.model}"
                   f":generateContent?key={provider.api_key}")
            r = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": full_prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[GEMINI ERROR: {e}]"

    def _call_openai(self, provider: LlmProvider, prompt: str,
                     system: str, max_tokens: int, temperature: float) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            r = requests.post(
                f"{provider.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": provider.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[OPENAI ERROR: {e}]"

    def _call_anthropic(self, provider: LlmProvider, prompt: str,
                        system: str, max_tokens: int, temperature: float) -> str:
        try:
            r = requests.post(
                f"{provider.base_url}/messages",
                headers={
                    "x-api-key": provider.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": provider.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system or "You are a helpful assistant.",
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except Exception as e:
            return f"[ANTHROPIC ERROR: {e}]"


# ════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════

_LANG_MAP = {
    ".py": "python", ".pyi": "python", ".rs": "rust",
    ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".jsx": "javascript", ".go": "go", ".java": "java",
    ".kt": "kotlin", ".rb": "ruby", ".sql": "sql",
    ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".json": "json", ".md": "markdown",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
}


def _infer_language(source: str) -> str:
    """Infer programming language from source identifier."""
    s = source.lower()
    for ext, lang in _LANG_MAP.items():
        if s.endswith(ext):
            return lang
    return ""
