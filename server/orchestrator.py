"""
Orchestrator — Dreamer Loop + Agent Lifecycle + Full Pipeline

Final stage of the cognitive pipeline. Runs the Dreamer cognitive cycle:
  Encode → Predict → Plan → Act → Observe → Surprise → Learn → Remember

This module connects ALL components:
  Commander    →  intent classification
  Planner      →  task decomposition
  Rust Kernel  →  persona, memory, world model, evolution, security, autotune

The Dreamer Loop (per agent):
  ┌── 1. ENCODE   → World Model latent state
  │   2. PREDICT  → Dynamics MLP: state_{t+1}
  │   3. PLAN     → Mental simulation: rollout N actions
  │   4. ACT      → LLM call (Entroly-optimized)
  │   5. OBSERVE  → Record actual outcome
  │   6. SURPRISE → S = (1 - cos(predicted, actual)) / 2
  │        ├─ S > 0.5 → broadcast via AMP
  │        ├─ S > 0.7 → re-plan
  │        └─ S < 0.1 → speculative pre-execute
  │   7. LEARN    → Update dynamics model, REINFORCE
  └── 8. REMEMBER → CLS consolidation (Ebbinghaus × surprise)
"""

from __future__ import annotations

import json
import math
import time
import uuid
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from commander import Commander, IntentResult
from planner import Planner, TaskPlan, TaskStep, StepStatus

# ── Rust kernel imports ──
try:
    import agentOS_kernel as kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  LOD TIERS — Level of Detail (from ebbiforge)
# ════════════════════════════════════════════════════════════════════

class LodTier(Enum):
    """Agent complexity tiers — only promote when needed."""
    DORMANT = 0      # Sleeping, zero cost
    SIMPLIFIED = 1   # Quick heuristic response
    FULL = 2         # Full pipeline, no LLM
    HEAVY = 3        # Full pipeline + LLM calls


def select_lod(intent: IntentResult) -> LodTier:
    """Select LOD tier based on intent classification."""
    if intent.action_class == "read_only" and intent.vagueness < 0.3:
        return LodTier.SIMPLIFIED     # Simple lookup, no LLM needed
    elif intent.needs_llm:
        return LodTier.HEAVY          # Complex task, needs LLM
    else:
        return LodTier.FULL           # Full pipeline, math-only


# ════════════════════════════════════════════════════════════════════
#  AGENT RUNTIME — One agent running the Dreamer loop
# ════════════════════════════════════════════════════════════════════

@dataclass
class AgentRuntime:
    """Runtime state of a single agent executing a plan."""
    agent_id: str
    persona_id: str
    lod_tier: LodTier
    plan: TaskPlan
    current_step_idx: int = 0
    total_surprise: float = 0.0
    total_steps_executed: int = 0
    re_plans: int = 0
    created_at: float = field(default_factory=time.time)

    def is_done(self) -> bool:
        return self.plan.is_complete()


# ════════════════════════════════════════════════════════════════════
#  DREAMER LOOP RESULT
# ════════════════════════════════════════════════════════════════════

@dataclass
class DreamerResult:
    """Result of one Dreamer loop iteration."""
    step_id: str
    description: str
    action_taken: str
    predicted_outcome: List[float]   # World model prediction
    actual_outcome: List[float]      # Observed result
    surprise: float                  # (1 - cosine_sim) / 2
    action_taken_by: str             # "math" or "llm"
    should_replan: bool              # S > 0.7
    should_broadcast: bool           # S > 0.5


# ════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR — The full pipeline
# ════════════════════════════════════════════════════════════════════

class Orchestrator:
    """
    Agent OS Orchestrator — connects all modules into the Dreamer loop.

    Pipeline:
      1. Commander classifies intent (no LLM)
      2. Planner decomposes into DAG (1 LLM call)
      3. Orchestrator runs Dreamer loop per step:
         Encode → Predict → Act → Observe → Surprise → Learn → Remember
      4. Security gate enforces money previews
      5. Autotune learns from outcomes

    This is the only class that touches the LLM. Everything else is Rust math.
    """

    def __init__(self, embedding_dim: int = 64, llm_fn: Optional[Callable] = None):
        self.embedding_dim = embedding_dim
        self.llm_fn = llm_fn  # External: (prompt: str) -> str

        # ── Initialize subsystems ──
        self.commander = Commander(embedding_dim=embedding_dim)
        self.planner = Planner(embedding_dim=embedding_dim)

        # Initialize Rust modules
        if KERNEL_AVAILABLE:
            self.world_model = kernel.WorldModel(
                embedding_dim,  # action_dim
                1024,           # trajectory_capacity
                0.001,          # learning_rate
            )
            self._tick = 0
            self.memory = kernel.MemoryManager(
                l1_budget=4096, l2_budget=16384,
                l3_budget=65536, half_life=50.0,
            )
            self.security_gate = kernel.SecurityGate(
                0.5,  # financial_sensitivity
                0.0,  # auto_approve_limit
            )
            self.autotune = kernel.AutotuneEngine(
                ema_alpha=0.3, drift_weight=0.1,
            )
            self.evolution = kernel.EvolutionEngine(
                0.15,  # mutation_sigma
                0.20,  # elite_fraction
                10,    # min_population
            )
            # ── New V4 modules ──
            self.cognitive_bus = kernel.CognitiveBus(
                1000,   # history_capacity
                0.05,   # global_alpha
            )
            self.context_pipeline = kernel.ContextPipeline(
                budget=4096, dedup_threshold=3,
            )
            self.protocol_bridge = kernel.ProtocolBridge(
                0.30,  # match_threshold
            )
            self.amp_kernel = kernel.AmpKernel()
        else:
            self.world_model = None
            self.memory = None
            self.security_gate = None
            self.autotune = None
            self.evolution = None
            self.cognitive_bus = None
            self.context_pipeline = None
            self.protocol_bridge = None
            self.amp_kernel = None

        # Subscribe orchestrator systems to the bus
        if self.cognitive_bus is not None:
            self.cognitive_bus.subscribe(
                "orchestrator", [], 256, 0.05  # wildcard: all events
            )
            self.cognitive_bus.subscribe(
                "security",
                ["security_alert", "compliance_blocked", "money_gate_triggered"],
                64, 0.10,
            )
            self.cognitive_bus.subscribe(
                "autotune",
                ["task_complete", "surprise_detected", "autotune_update"],
                32, 0.05,
            )

        # Active agent runtimes
        self.active_agents: Dict[str, AgentRuntime] = {}
        self.completed_tasks: List[Dict[str, Any]] = []

        # Stats
        self.total_tasks = 0
        self.total_llm_calls = 0
        self.total_math_steps = 0
        self.total_surprises = 0.0

    # ════════════════════════════════════════════════════════════════
    #  MAIN ENTRY POINT
    # ════════════════════════════════════════════════════════════════

    def process_command(self, command: str,
                        embedding: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Process a user command through the full pipeline.

        Returns a dict with the plan, current status, and any previews needed.
        """
        self.total_tasks += 1

        # ── Step 1: Commander — classify intent (no LLM) ──
        intent = self.commander.classify(command, embedding)

        # Publish intent_classified event to ISA-Bus
        if self.cognitive_bus is not None:
            self.cognitive_bus.publish(
                "commander", "intent_classified",
                intent.persona_affinity,
                json.dumps({"domain": intent.domain, "vagueness": intent.vagueness}),
            )

        # ── Step 2: Select LOD tier ──
        lod = select_lod(intent)

        # ── Step 3: Protocol routing — check if external capability needed ──
        protocol_match = None
        if self.protocol_bridge is not None:
            protocol_match = self.protocol_bridge.route(
                command, "a2a", intent.action_class,
            )

        # ── Step 4: Planner — decompose into DAG ──
        plan = self.planner.plan(
            command=command,
            domain=intent.domain,
            persona_id=intent.persona_id,
            action_class=intent.action_class,
        )

        # ── Step 5: Create agent runtime ──
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        runtime = AgentRuntime(
            agent_id=agent_id,
            persona_id=intent.persona_id,
            lod_tier=lod,
            plan=plan,
        )
        self.active_agents[agent_id] = runtime

        # Register agent in AMP kernel for multi-agent negotiation
        if self.amp_kernel is not None:
            agent_intent = (intent.embedding or [0.0] * 7)[:7]
            while len(agent_intent) < 7:
                agent_intent.append(0.0)
            self.amp_kernel.register_agent(agent_id, agent_intent)

        # Register agent in evolution engine
        if self.evolution is not None:
            self.evolution.register_agent(agent_id, True)

        # ── Step 6: Run ready steps ──
        results = self._execute_ready_steps(runtime)

        # ── Step 7: Build response ──
        response = {
            "agent_id": agent_id,
            "command": command,
            "intent": {
                "domain": intent.domain,
                "vagueness": intent.vagueness,
                "persona_id": intent.persona_id,
                "persona_is_new": intent.persona_is_new,
                "persona_affinity": intent.persona_affinity,
                "action_class": intent.action_class,
                "needs_llm": intent.needs_llm,
            },
            "lod_tier": lod.name,
            "plan": plan.to_dict(),
            "step_results": [r.__dict__ for r in results],
            "needs_approval": len(plan.get_blocked_steps()) > 0,
            "blocked_steps": [s.to_dict() for s in plan.get_blocked_steps()],
            "is_complete": plan.is_complete(),
        }

        # Add protocol routing info if available
        if protocol_match is not None:
            response["protocol_route"] = protocol_match

        # Remember the task
        if self.memory is not None:
            self.memory.remember(
                0, f"Task: {command} → domain={intent.domain}, "
                   f"persona={intent.persona_id}",
                intent.persona_affinity, "working", None,
            )

        return response

    def approve_step(self, agent_id: str, step_id: str) -> Dict[str, Any]:
        """User approves a money gate step. Continues execution."""
        runtime = self.active_agents.get(agent_id)
        if not runtime:
            return {"error": f"No active agent: {agent_id}"}

        if self.planner.approve_step(runtime.plan, step_id):
            results = self._execute_ready_steps(runtime)
            return {
                "approved": True,
                "step_id": step_id,
                "step_results": [r.__dict__ for r in results],
                "is_complete": runtime.plan.is_complete(),
            }
        return {"error": f"Step {step_id} not found or not in preview state"}

    def reject_step(self, agent_id: str, step_id: str) -> Dict[str, Any]:
        """User rejects a money gate step. Skips downstream."""
        runtime = self.active_agents.get(agent_id)
        if not runtime:
            return {"error": f"No active agent: {agent_id}"}

        if self.planner.reject_step(runtime.plan, step_id):
            return {
                "rejected": True,
                "step_id": step_id,
                "is_complete": runtime.plan.is_complete(),
            }
        return {"error": f"Step {step_id} not found or not in preview state"}

    # ════════════════════════════════════════════════════════════════
    #  DREAMER LOOP (per step)
    # ════════════════════════════════════════════════════════════════

    def _execute_ready_steps(self, runtime: AgentRuntime) -> List[DreamerResult]:
        """Execute all ready steps through the Dreamer loop."""
        results = []

        while True:
            ready_steps = runtime.plan.get_ready_steps()
            if not ready_steps:
                break

            for step in ready_steps:
                # Check if preview is needed (money gate)
                if step.action_class == "money_out" and step.user_approved is None:
                    step.status = StepStatus.PREVIEW
                    continue

                result = self._dreamer_step(runtime, step)
                results.append(result)

                runtime.total_steps_executed += 1
                runtime.total_surprise += result.surprise

                # Surprise-driven re-planning
                if result.should_replan:
                    runtime.re_plans += 1
                    # In production: call LLM to generate new plan
                    break

        # If done, record completion
        if runtime.plan.is_complete():
            self._record_completion(runtime)

        return results

    def _dreamer_step(self, runtime: AgentRuntime, step: TaskStep) -> DreamerResult:
        """
        Run one Dreamer loop iteration for a single step.

        1. ENCODE   → latent state
        2. PREDICT  → expected outcome
        3. ACT      → execute (math or LLM)
        4. OBSERVE  → actual outcome
        5. SURPRISE → prediction error
        6. LEARN    → update world model
        7. REMEMBER → store in memory
        """
        step.status = StepStatus.RUNNING

        # ── 1. ENCODE ──
        step_emb = self._embed_step(step)
        latent = None
        if self.world_model is not None:
            try:
                self._tick += 1
                latent = self.world_model.encode(step_emb, self._tick)
            except Exception:
                latent = None

        # ── 2. PREDICT ──
        predicted = step_emb[:]  # Fallback: predict same as input
        if self.world_model is not None and latent is not None:
            try:
                predicted, _ = self.world_model.predict(step_emb)
            except Exception:
                pass

        # ── 3. ACT ──
        action_by = "math"
        action_result = f"Executed: {step.description}"

        if runtime.lod_tier == LodTier.HEAVY and self.llm_fn is not None:
            # LLM call
            try:
                persona_prompt = f"You are persona {runtime.persona_id}. "
                action_result = self.llm_fn(
                    persona_prompt + step.description
                )
                action_by = "llm"
                self.total_llm_calls += 1
            except Exception as e:
                action_result = f"LLM error: {e}"
        else:
            self.total_math_steps += 1

        # ── 4. OBSERVE ──
        actual = self._embed_step(step)  # In production: embed actual outcome

        # ── 5. SURPRISE ──
        surprise = self._cosine_surprise(predicted, actual)
        step.surprise = surprise
        self.total_surprises += surprise

        # Publish surprise event to ISA-Bus
        if self.cognitive_bus is not None:
            event_type = "surprise_detected" if surprise > 0.3 else "task_progress"
            self.cognitive_bus.publish(
                runtime.persona_id, event_type, surprise,
                json.dumps({"step": step.description[:50], "surprise": round(surprise, 4)}),
            )

        # ── 6. LEARN ──
        if self.world_model is not None and latent is not None:
            try:
                self.world_model.observe(predicted, actual, step_emb)
            except Exception:
                pass

        # Update evolution fitness based on surprise
        if self.evolution is not None:
            self.evolution.update_fitness(
                runtime.agent_id, surprise < 0.3, surprise,
            )

        # ── 7. REMEMBER ──
        if self.memory is not None:
            salience = max(0.3, surprise)  # Higher surprise = more memorable
            self.memory.remember(
                0,
                f"Step '{step.description}' → surprise={surprise:.3f}: {action_result[:100]}",
                salience,
                "working" if surprise < 0.3 else "episodic",
                None,
            )

        step.status = StepStatus.COMPLETED
        step.result = action_result

        return DreamerResult(
            step_id=step.id,
            description=step.description,
            action_taken=action_result[:200],
            predicted_outcome=predicted[:8],  # Truncate for readability
            actual_outcome=actual[:8],
            surprise=surprise,
            action_taken_by=action_by,
            should_replan=surprise > 0.7,
            should_broadcast=surprise > 0.5,
        )

    # ════════════════════════════════════════════════════════════════
    #  LIFECYCLE + LEARNING
    # ════════════════════════════════════════════════════════════════

    def tick(self):
        """Run one lifecycle tick — memory, personas, bus, evolution, autotune."""
        if self.memory is not None:
            self.memory.tick()

        # Persona lifecycle: decay, death, fusion
        self.commander.lifecycle_tick()

        # ISA-Bus tick: update rate models, compute KL priorities
        if self.cognitive_bus is not None:
            self.cognitive_bus.tick()

        # Evolution tick
        if self.evolution is not None:
            self.evolution.evolve()

    def consolidate(self):
        """Promote important memories from L1→L2→L3."""
        if self.memory is not None:
            self.memory.consolidate(0.7, 2)

    def forget(self, threshold: float = 0.05) -> int:
        """Apply Ebbinghaus decay and forget low-importance memories."""
        if self.memory is not None:
            return self.memory.forget(threshold)
        return 0

    # ════════════════════════════════════════════════════════════════
    #  STATS + STATE
    # ════════════════════════════════════════════════════════════════

    def stats(self) -> Dict[str, Any]:
        """Full orchestrator stats."""
        result = {
            "total_tasks": self.total_tasks,
            "total_llm_calls": self.total_llm_calls,
            "total_math_steps": self.total_math_steps,
            "math_ratio": (self.total_math_steps /
                          max(self.total_math_steps + self.total_llm_calls, 1)),
            "total_surprises": self.total_surprises,
            "active_agents": len(self.active_agents),
            "completed_tasks": len(self.completed_tasks),
            "kernel_available": KERNEL_AVAILABLE,
            "commander": self.commander.stats(),
            "planner": self.planner.stats(),
        }
        if self.memory is not None:
            result["memory"] = self.memory.stats()
        if self.autotune is not None:
            result["autotune"] = self.autotune.stats()
        if self.cognitive_bus is not None:
            result["cognitive_bus"] = self.cognitive_bus.stats()
        if self.protocol_bridge is not None:
            result["protocol_bridge"] = self.protocol_bridge.stats()
        if self.amp_kernel is not None:
            result["amp_kernel"] = self.amp_kernel.stats()
        return result

    # ── Internal helpers ──

    def _embed_step(self, step: TaskStep) -> List[float]:
        """Create a simple embedding for a step. In production: use fastembed."""
        emb = [0.0] * self.embedding_dim
        for i, ch in enumerate(step.description):
            emb[i % self.embedding_dim] += ord(ch) / 500.0
        norm = math.sqrt(sum(x * x for x in emb)) or 1.0
        return [x / norm for x in emb]

    def _cosine_surprise(self, predicted: List[float],
                         actual: List[float]) -> float:
        """Surprise = (1 - cosine_similarity) / 2."""
        if len(predicted) != len(actual):
            return 0.5

        dot = sum(a * b for a, b in zip(predicted, actual))
        norm_p = math.sqrt(sum(x * x for x in predicted)) or 1.0
        norm_a = math.sqrt(sum(x * x for x in actual)) or 1.0
        cosine = dot / (norm_p * norm_a)
        return (1.0 - cosine) / 2.0

    def _record_completion(self, runtime: AgentRuntime):
        """Record task completion for learning."""
        avg_surprise = (runtime.total_surprise /
                       max(runtime.total_steps_executed, 1))
        success = avg_surprise < 0.3  # Low surprise = good persona match

        record = {
            "agent_id": runtime.agent_id,
            "persona_id": runtime.persona_id,
            "lod_tier": runtime.lod_tier.name,
            "total_steps": runtime.total_steps_executed,
            "total_surprise": runtime.total_surprise,
            "re_plans": runtime.re_plans,
            "duration": time.time() - runtime.created_at,
            "command": runtime.plan.command,
            "success": success,
        }
        self.completed_tasks.append(record)

        # Record result for persona learning
        embedding = self._embed_step(runtime.plan.steps[0]) if runtime.plan.steps else [0.0] * self.embedding_dim
        self.commander.record_result(runtime.persona_id, success, embedding)

        # Publish completion event to ISA-Bus
        if self.cognitive_bus is not None:
            self.cognitive_bus.publish(
                runtime.persona_id, "task_complete", avg_surprise,
                json.dumps({"agent": runtime.agent_id, "success": success,
                            "steps": runtime.total_steps_executed}),
            )

        # Clean up
        if runtime.agent_id in self.active_agents:
            del self.active_agents[runtime.agent_id]
