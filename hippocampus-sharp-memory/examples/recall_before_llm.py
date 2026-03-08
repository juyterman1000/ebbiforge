#!/usr/bin/env python3
"""
Recall-Before-LLM Pattern
==========================

The business-critical use case for hippocampus-memory:
save 90%+ on LLM API costs by checking memory before calling the LLM.

How it works:
  1. Alert comes in
  2. Check if we've explained a similar alert before
  3. If cached → return instantly (free!)
  4. If new → call LLM, cache the response for next time

The Ebbinghaus forgetting curve naturally evicts stale explanations,
so cached responses don't go stale forever.
"""

import time
from hippocampus_sharp_memory import create_memory


def simulate_llm_call(prompt: str) -> str:
    """Simulate an LLM API call (200ms latency, $0.003/call)."""
    time.sleep(0.2)  # Simulate network latency
    return f"Analysis: {prompt[:50]} — this is caused by normal load patterns."


class SmartAlertHandler:
    """Alert handler with hippocampus-memory caching."""

    def __init__(self):
        self.mem = create_memory(capacity=100_000)
        self.llm_calls = 0
        self.cache_hits = 0

    def handle(self, alert_text: str) -> str:
        # Step 1: Check memory for a cached explanation
        cached = self.mem.recall(alert_text, top_k=1)

        if cached and cached[0].retention > 0.3:
            self.cache_hits += 1
            return cached[0].content

        # Step 2: Cache miss — call the LLM
        self.llm_calls += 1
        explanation = simulate_llm_call(alert_text)

        # Step 3: Store the expensive response in memory
        self.mem.remember(
            explanation,
            salience=60.0,
            source="llm_cache",
        )

        return explanation

    @property
    def hit_rate(self) -> float:
        total = self.llm_calls + self.cache_hits
        return self.cache_hits / total * 100 if total > 0 else 0.0


# ── Simulation ─────────────────────────────────────────────────────────
print("🧠 Recall-Before-LLM Pattern Demo\n")

handler = SmartAlertHandler()

# Simulate a realistic alert stream: most alerts are recurring
alerts = [
    # First occurrence of each alert type (will call LLM)
    "CPU spike to 95% on web-server-01",
    "Memory usage exceeded 80% threshold",
    "Disk I/O latency above 50ms",
    "SSL certificate expiring in 7 days",
    "Rate limit exceeded for API endpoint /v1/users",
    # Recurring alerts (should hit cache)
    "CPU spike to 95% on web-server-01",
    "CPU spike to 95% on web-server-01",
    "Memory usage exceeded 80% threshold",
    "CPU spike to 95% on web-server-01",
    "Disk I/O latency above 50ms",
    "Memory usage exceeded 80% threshold",
    "Rate limit exceeded for API endpoint /v1/users",
    "CPU spike to 95% on web-server-01",
    "SSL certificate expiring in 7 days",
    "Memory usage exceeded 80% threshold",
    "CPU spike to 95% on web-server-01",
    "Disk I/O latency above 50ms",
    "Rate limit exceeded for API endpoint /v1/users",
    "CPU spike to 95% on web-server-01",
    "Memory usage exceeded 80% threshold",
]

t0 = time.perf_counter()

for i, alert in enumerate(alerts, 1):
    result = handler.handle(alert)
    source = "💰 CACHE" if handler.cache_hits > (i - handler.llm_calls - 1) else "🔥 LLM  "
    # Simple detection: if llm_calls didn't change, it was a cache hit
    print(f"  [{i:2d}] {alert[:50]}")

elapsed = time.perf_counter() - t0

print(f"\n{'═' * 60}")
print(f"  Total alerts processed: {len(alerts)}")
print(f"  LLM calls made:        {handler.llm_calls}")
print(f"  Cache hits:             {handler.cache_hits}")
print(f"  Hit rate:               {handler.hit_rate:.0f}%")
print(f"  Time elapsed:           {elapsed:.1f}s")
print(f"{'═' * 60}")

# Cost analysis
llm_cost_per_call = 0.003  # $0.003 per GPT-4-mini call
without_cache = len(alerts) * llm_cost_per_call
with_cache = handler.llm_calls * llm_cost_per_call
savings = without_cache - with_cache
print(f"\n  💰 Cost without cache: ${without_cache:.3f}")
print(f"  💰 Cost with cache:   ${with_cache:.3f}")
print(f"  💰 Savings:           ${savings:.3f} ({savings/without_cache*100:.0f}%)")

print("\n✅ Memory that pays for itself.")
