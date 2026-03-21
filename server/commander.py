"""
Commander — Intent Classification + Domain Routing

First stage of the cognitive pipeline:
  User Command → Parse → Classify → Route to Persona → Return Intent

Pipeline (no LLM until step 4):
  1. TF-IDF keyword extraction (Rust math)
  2. Vagueness scoring (heuristic)
  3. PSM persona selection (Rust: PersonaManifold.assign())
  4. LLM intent classification (1 small call, persona-informed)

This module bridges:
  agentOS_kernel.PersonaManifold  →  Python orchestration
  agentOS_kernel.SecurityGate     →  action classification
  agentOS_kernel.MemoryManager    →  context retrieval
"""

from __future__ import annotations

import re
import math
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# ── Rust kernel imports (graceful fallback) ──
try:
    import agentOS_kernel as kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════
#  TF-IDF KEYWORD EXTRACTION (pure Python — fast, no LLM)
# ════════════════════════════════════════════════════════════════════

# Domain keyword dictionaries for TF-IDF scoring
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance": ["pay", "bill", "transfer", "money", "account", "balance",
                "invoice", "bank", "credit", "debit", "payment", "budget",
                "expense", "refund", "subscribe", "cancel"],
    "shopping": ["buy", "order", "purchase", "shop", "cart", "deal",
                 "price", "compare", "coupon", "discount", "amazon",
                 "delivery", "shipping", "return"],
    "health": ["doctor", "appointment", "prescription", "pharmacy",
               "health", "medical", "dentist", "insurance", "hospital",
               "symptom", "medicine", "checkup", "lab"],
    "travel": ["flight", "hotel", "book", "reservation", "trip",
               "airline", "passport", "visa", "itinerary", "uber",
               "lyft", "rental", "vacation"],
    "productivity": ["email", "calendar", "meeting", "schedule", "remind",
                     "task", "todo", "note", "organize", "plan",
                     "document", "spreadsheet", "presentation"],
    "tech": ["code", "deploy", "debug", "server", "database", "api",
             "git", "build", "test", "install", "update", "backup",
             "configure", "monitor", "log"],
    "home": ["electric", "gas", "water", "utility", "internet", "phone",
             "maintenance", "repair", "clean", "grocery", "food",
             "recipe", "cook", "laundry"],
    "social": ["message", "call", "text", "invite", "rsvp", "gift",
               "birthday", "party", "group", "family", "friend"],
}

# Stopwords for vagueness scoring
STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "being", "have", "has", "had", "do", "does", "did", "will",
             "would", "could", "should", "may", "might", "shall", "can",
             "i", "me", "my", "we", "our", "you", "your", "it", "its",
             "they", "them", "their", "this", "that", "these", "those",
             "to", "of", "in", "for", "on", "with", "at", "by", "from",
             "and", "or", "but", "not", "so", "if", "then", "just",
             "about", "up", "out", "some", "what", "how", "when", "where"}


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return [w.lower().strip(".,!?;:'\"()[]{}") for w in text.split() if w]


def tf_idf_domains(text: str) -> Dict[str, float]:
    """Score each domain by TF-IDF of its keywords in the text."""
    tokens = set(tokenize(text))
    scores: Dict[str, float] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        # TF: fraction of domain keywords found in text
        hits = sum(1 for k in keywords if k in tokens)
        tf = hits / max(len(keywords), 1)
        # IDF: rarer domains get boosted (more keywords = more specific)
        idf = math.log(1 + len(DOMAIN_KEYWORDS) / max(1, sum(
            1 for d, kws in DOMAIN_KEYWORDS.items()
            if any(k in tokens for k in kws)
        )))
        scores[domain] = tf * idf

    return scores


def vagueness_score(text: str) -> float:
    """Score [0, 1] how vague the command is. High = needs clarification."""
    tokens = tokenize(text)
    if not tokens:
        return 1.0

    content_words = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    content_ratio = len(content_words) / max(len(tokens), 1)

    # Short commands are more vague
    length_penalty = max(0, 1.0 - len(tokens) / 8.0)

    # Question words increase specificity
    question_words = {"what", "how", "when", "where", "who", "which", "why"}
    has_question = any(t in question_words for t in tokens)

    vagueness = (1.0 - content_ratio) * 0.5 + length_penalty * 0.3
    if has_question:
        vagueness *= 0.7  # Questions are more specific

    return min(max(vagueness, 0.0), 1.0)


# ════════════════════════════════════════════════════════════════════
#  INTENT RESULT
# ════════════════════════════════════════════════════════════════════

@dataclass
class IntentResult:
    """Result of intent classification."""
    raw_command: str
    domain: str                     # Top domain from TF-IDF
    domain_scores: Dict[str, float] # All domain scores
    vagueness: float                # [0, 1]
    persona_id: str                 # PSM-assigned persona
    persona_is_new: bool            # Whether persona was born
    persona_affinity: float         # PSM affinity score
    action_class: str               # Security classification
    needs_llm: bool                 # Whether LLM is needed for classification
    embedding: List[float]          # Task embedding (for PSM)
    timestamp: float = field(default_factory=time.time)


# ════════════════════════════════════════════════════════════════════
#  COMMANDER
# ════════════════════════════════════════════════════════════════════

class Commander:
    """
    Intent classification + domain routing + persona selection.

    Pipeline (3 stages before any LLM call):
      1. TF-IDF domain scoring (pure Python, ~1ms)
      2. Vagueness scoring (heuristic, ~0.1ms)
      3. PSM persona selection (Rust, ~50μs)
      4. [Optional] LLM intent refinement
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

        # Initialize Rust kernel modules
        if KERNEL_AVAILABLE:
            self.persona_manifold = kernel.PersonaManifold(
                dim=embedding_dim, alpha=1.0, discount=0.25,
                fusion_threshold=0.01
            )
            self.security_gate = kernel.SecurityGate(
                0.5,  # financial_sensitivity
                0.0,  # auto_approve_limit
            )
            self.memory = kernel.MemoryManager(
                l1_budget=4096, l2_budget=16384,
                l3_budget=65536, half_life=50.0
            )
        else:
            self.persona_manifold = None
            self.security_gate = None
            self.memory = None

        self.total_commands = 0
        self.total_births = 0

    def classify(self, command: str, embedding: Optional[List[float]] = None) -> IntentResult:
        """
        Classify a user command through the full pipeline.

        Returns IntentResult with domain, persona, action class, and
        whether LLM is needed for further refinement.
        """
        self.total_commands += 1

        # ── Stage 1: TF-IDF domain scoring ──
        domain_scores = tf_idf_domains(command)
        top_domain = max(domain_scores, key=domain_scores.get) if domain_scores else "general"
        top_score = domain_scores.get(top_domain, 0.0)

        # ── Stage 2: Vagueness scoring ──
        vagueness = vagueness_score(command)

        # ── Stage 3: PSM persona selection ──
        if embedding is None:
            embedding = self._simple_embedding(command)

        persona_id = "fallback"
        persona_is_new = False
        persona_affinity = 0.0

        if self.persona_manifold is not None:
            persona_id, persona_affinity, persona_is_new = \
                self.persona_manifold.assign(embedding)
            if persona_is_new:
                self.total_births += 1

        # ── Stage 4: Action classification ──
        action_class = self._classify_action(command, top_domain)

        # ── Determine if LLM is needed ──
        needs_llm = (
            top_score < 0.1 or        # No strong domain match
            vagueness > 0.6 or         # Too vague
            action_class == "money_out" # Financial needs confirmation
        )

        return IntentResult(
            raw_command=command,
            domain=top_domain,
            domain_scores=domain_scores,
            vagueness=vagueness,
            persona_id=persona_id,
            persona_is_new=persona_is_new,
            persona_affinity=persona_affinity,
            action_class=action_class,
            needs_llm=needs_llm,
            embedding=embedding,
        )

    def record_result(self, persona_id: str, success: bool,
                       embedding: List[float]) -> None:
        """Record task outcome for persona learning."""
        if self.persona_manifold is not None:
            self.persona_manifold.record_result(persona_id, success, embedding)

    def lifecycle_tick(self) -> str:
        """Run persona lifecycle (decay, death, fusion)."""
        if self.persona_manifold is not None:
            return self.persona_manifold.lifecycle_tick()
        return "[]"

    def stats(self) -> Dict[str, Any]:
        """Get commander statistics."""
        result = {
            "total_commands": self.total_commands,
            "total_births": self.total_births,
            "kernel_available": KERNEL_AVAILABLE,
        }
        if self.persona_manifold is not None:
            result["persona_stats"] = self.persona_manifold.stats()
        return result

    # ── Internal helpers ──

    def _simple_embedding(self, text: str) -> List[float]:
        """Generate a simple TF-IDF-based embedding for PSM.
        In production, use fastembed or OpenAI embeddings."""
        emb = [0.0] * self.embedding_dim
        tokens = tokenize(text)

        for i, (domain, keywords) in enumerate(DOMAIN_KEYWORDS.items()):
            if i >= self.embedding_dim:
                break
            hits = sum(1 for k in keywords if k in set(tokens))
            emb[i] = hits / max(len(keywords), 1)

        # Character hash features for remaining dimensions
        for j, ch in enumerate(text.lower()):
            idx = (j + len(DOMAIN_KEYWORDS)) % self.embedding_dim
            emb[idx] += ord(ch) / 1000.0

        # Normalize
        norm = math.sqrt(sum(x * x for x in emb)) or 1.0
        return [x / norm for x in emb]

    def _classify_action(self, command: str, domain: str) -> str:
        """Classify action type from command text and domain."""
        cmd_lower = command.lower()

        # Money indicators
        money_words = {"pay", "transfer", "send", "buy", "purchase",
                       "subscribe", "donate", "tip", "charge"}
        if any(w in cmd_lower.split() for w in money_words):
            return "money_out"

        # Destructive indicators
        destroy_words = {"delete", "remove", "destroy", "cancel", "terminate",
                         "revoke", "unsubscribe", "close account"}
        if any(w in cmd_lower for w in destroy_words):
            return "irreversible"

        # Read-only indicators
        read_words = {"check", "show", "list", "view", "find", "search",
                      "look up", "what", "how", "when", "where", "status"}
        if any(w in cmd_lower for w in read_words):
            return "read_only"

        # Default: reversible
        return "reversible"
