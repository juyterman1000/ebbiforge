"""
Hippocampus Sharp Memory — Brain-Inspired Memory for AI Agents
=========================================================

O(1) recall via LSH index, adaptive forgetting, context-weighted
scoring, and optional disk-backed persistence with user-controlled quotas.

Salience Guide (lifetime = salience * 4.6 ticks)::

    salience=1     ~5 ticks      throwaway (greetings, acks)
    salience=10    ~46 ticks     session context (current topic)
    salience=50    ~230 ticks    important facts (user prefs)
    salience=100   ~460 ticks    business-critical (complaints)
    salience=500   ~2300 ticks   near-permanent (compliance)

Quick Start::

    from hippocampus_sharp_memory import create_memory

    mem = create_memory()
    mem.remember("user prefers dark mode", salience=50.0)     # ~230 ticks
    mem.remember("routine greeting", salience=1.0)             # ~5 ticks
    mem.tick()  # call per second, per message, or per event

    results = mem.recall("dark mode", top_k=3)
    for r in results:
        print(f"  [{r.retention*100:.0f}% retained] {r.content}")

Part of the `Ebbiforge <https://github.com/juyterman1000/ebbiforge>`_ ecosystem.

.. note::
    This package is a thin wrapper around ``ebbiforge_core.HippocampusEngine``
    (written in Rust via PyO3). Zero code duplication — same engine, same speed.
"""

__version__ = "1.0.3"
__all__ = [
    # Core engine
    "HippocampusEngine",
    "MemoryBankConfig",
    # Result types
    "RecallResult",
    "HippocampusStats",
    "Episode",
    # Factory functions
    "create_memory",
    "create_persistent_memory",
]

# ── Import from Rust core ─────────────────────────────────────────────────
try:
    from ebbiforge_core import (
        HippocampusEngine,
        MemoryBankConfig,
        RecallResult,
        HippocampusStats,
        Episode,
    )
except ImportError:
    raise ImportError(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  hippocampus-sharp-memory requires the Ebbiforge Rust      ║\n"
        "║  engine (ebbiforge-core).                                  ║\n"
        "║                                                            ║\n"
        "║  Install it:                                               ║\n"
        "║    pip install ebbiforge                                    ║\n"
        "║                                                            ║\n"
        "║  Or build from source:                                     ║\n"
        "║    git clone https://github.com/juyterman1000/ebbiforge    ║\n"
        "║    cd ebbiforge && pip install maturin                     ║\n"
        "║    maturin develop --release                               ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    ) from None


# ── Factory Functions ──────────────────────────────────────────────────────

def create_memory(
    capacity: int = 500_000,
    consolidation_interval: int = 100,
    recall_reinforcement: float = 1.3,
) -> HippocampusEngine:
    """Create an in-memory HippocampusEngine with sensible defaults.

    This is the fastest way to get started. Memories live in RAM only
    and are lost when the process exits.

    Args:
        capacity: Maximum number of episodes before eviction (default: 500K).
        consolidation_interval: Ticks between sleep-replay consolidation cycles.
        recall_reinforcement: Salience multiplier on each recall (spaced repetition).

    Returns:
        A configured ``HippocampusEngine`` ready to use.

    Example::

        mem = create_memory()
        mem.remember("user prefers dark mode", salience=30.0)
        results = mem.recall("dark mode preference", top_k=1)
    """
    return HippocampusEngine(
        capacity=capacity,
        consolidation_interval=consolidation_interval,
        recall_reinforcement=recall_reinforcement,
    )


def create_persistent_memory(
    quota_gb: float = 7.5,
    capacity: int = 1_000_000,
    storage_path: str = "",
    consolidation_interval: int = 100,
    recall_reinforcement: float = 1.3,
) -> "HippocampusEngine":
    """Create a large-capacity HippocampusEngine optimized for long-running processes.

    Uses higher defaults than ``create_memory()`` — 1M episode capacity,
    faster consolidation. Ideal for production services that run for hours/days.

    .. note::
        Disk-backed persistence (via ``MemoryBankConfig``) is available in the
        full Ebbiforge framework. This function creates a high-capacity
        in-memory engine suitable for long-running processes.

    Args:
        quota_gb: Reserved for future disk-backed mode (not yet wired).
        capacity: Maximum in-memory episodes (default: 1M).
        storage_path: Reserved for future disk-backed mode.
        consolidation_interval: Ticks between consolidation cycles.
        recall_reinforcement: Salience multiplier on recall.

    Returns:
        A high-capacity ``HippocampusEngine``.

    Example::

        mem = create_persistent_memory()
        mem.remember("critical incident report", salience=90.0, emotional_tag=3)
    """
    return HippocampusEngine(
        capacity=capacity,
        consolidation_interval=consolidation_interval,
        recall_reinforcement=recall_reinforcement,
    )
