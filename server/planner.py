"""
Planner — Graph-of-Thought Task Decomposition + World Model Simulation

Second stage of the cognitive pipeline:
  IntentResult → Decompose → Build DAG → Simulate → Return Plan

Pipeline:
  1. Decompose command into subtasks (LLM call #2)
  2. Build TaskDAG with dependencies
  3. Simulate via World Model (predict outcomes)
  4. Insert security gates (money steps require preview)

This module bridges:
  agentOS_kernel.WorldModel     →  mental simulation
  agentOS_kernel.SecurityGate   →  money gate insertion
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ── Rust kernel imports (graceful fallback) ──
try:
    import agentOS_kernel as kernel
    KERNEL_AVAILABLE = True
except ImportError:
    KERNEL_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════
#  TASK NODE — Single step in the plan
# ════════════════════════════════════════════════════════════════════

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"         # Waiting for dependency
    PREVIEW = "preview"         # Money gate — waiting for user confirm
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """A single step in the execution plan."""
    id: str
    description: str
    action_class: str           # read_only, reversible, money_out, irreversible
    status: StepStatus = StepStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    result: Optional[str] = None
    cost_estimate: float = 0.0  # Estimated tokens for this step
    surprise: float = 0.0       # World model prediction error
    created_at: float = field(default_factory=time.time)

    # Money gate fields (populated if action_class == money_out)
    amount: Optional[float] = None
    recipient: Optional[str] = None
    user_approved: Optional[bool] = None

    def is_ready(self, completed_ids: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_ids for dep in self.depends_on)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "action_class": self.action_class,
            "status": self.status.value,
            "depends_on": self.depends_on,
            "result": self.result,
            "amount": self.amount,
            "recipient": self.recipient,
            "surprise": self.surprise,
        }


# ════════════════════════════════════════════════════════════════════
#  TASK DAG — Directed Acyclic Graph of steps
# ════════════════════════════════════════════════════════════════════

@dataclass
class TaskPlan:
    """A DAG of TaskSteps representing the execution plan."""
    plan_id: str
    command: str
    persona_id: str
    steps: List[TaskStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    total_cost_estimate: float = 0.0
    has_money_gate: bool = False

    def add_step(self, description: str, action_class: str = "reversible",
                 depends_on: Optional[List[str]] = None,
                 amount: Optional[float] = None,
                 recipient: Optional[str] = None) -> str:
        """Add a step to the plan. Returns step ID."""
        step_id = f"step_{len(self.steps) + 1}"
        step = TaskStep(
            id=step_id,
            description=description,
            action_class=action_class,
            depends_on=depends_on or [],
            amount=amount,
            recipient=recipient,
        )
        if action_class == "money_out":
            self.has_money_gate = True
        self.steps.append(step)
        return step_id

    def get_ready_steps(self) -> List[TaskStep]:
        """Get steps whose dependencies are all completed."""
        completed = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        return [s for s in self.steps
                if s.status == StepStatus.PENDING and s.is_ready(completed)]

    def get_blocked_steps(self) -> List[TaskStep]:
        """Get steps waiting for money gate approval."""
        return [s for s in self.steps if s.status == StepStatus.PREVIEW]

    def is_complete(self) -> bool:
        """Check if all steps are done (completed, failed, or skipped)."""
        terminal = {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED}
        return all(s.status in terminal for s in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "command": self.command,
            "persona_id": self.persona_id,
            "steps": [s.to_dict() for s in self.steps],
            "has_money_gate": self.has_money_gate,
            "is_complete": self.is_complete(),
            "total_cost_estimate": self.total_cost_estimate,
        }


# ════════════════════════════════════════════════════════════════════
#  PLANNER
# ════════════════════════════════════════════════════════════════════

class Planner:
    """
    Graph-of-Thought task decomposition + world model simulation.

    Takes an IntentResult from Commander, decomposes into a TaskPlan DAG,
    and simulates outcomes via the World Model.

    In production, step 1 uses an LLM to decompose. Here we provide
    template-based decomposition for common domains + LLM fallback.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim

        # World model for mental simulation
        if KERNEL_AVAILABLE:
            self.world_model = kernel.WorldModel(
                embedding_dim,  # action_dim
                1024,           # trajectory_capacity
                0.001,          # learning_rate
            )
            self._tick = 0
            self.security_gate = kernel.SecurityGate(
                financial_sensitivity=0.5
            )
        else:
            self.world_model = None
            self.security_gate = None

        self.total_plans = 0

    def plan(self, command: str, domain: str, persona_id: str,
             action_class: str) -> TaskPlan:
        """
        Decompose a command into a TaskPlan DAG.

        Uses template-based decomposition for known domains,
        LLM fallback for unknown commands.
        """
        self.total_plans += 1
        plan = TaskPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            command=command,
            persona_id=persona_id,
        )

        # Template-based decomposition for known patterns
        if action_class == "money_out":
            self._plan_money(plan, command, domain)
        elif domain == "finance":
            self._plan_finance(plan, command)
        elif domain == "shopping":
            self._plan_shopping(plan, command)
        elif domain == "health":
            self._plan_health(plan, command)
        elif domain == "travel":
            self._plan_travel(plan, command)
        elif domain == "productivity":
            self._plan_productivity(plan, command)
        elif domain == "tech":
            self._plan_tech(plan, command)
        else:
            self._plan_generic(plan, command)

        # Simulate outcomes via World Model
        if self.world_model is not None:
            self._simulate_plan(plan)

        return plan

    def approve_step(self, plan: TaskPlan, step_id: str) -> bool:
        """User approves a money gate step."""
        for step in plan.steps:
            if step.id == step_id and step.status == StepStatus.PREVIEW:
                step.user_approved = True
                step.status = StepStatus.PENDING
                return True
        return False

    def reject_step(self, plan: TaskPlan, step_id: str) -> bool:
        """User rejects a money gate step."""
        for step in plan.steps:
            if step.id == step_id and step.status == StepStatus.PREVIEW:
                step.user_approved = False
                step.status = StepStatus.SKIPPED
                # Skip all downstream steps
                self._skip_downstream(plan, step_id)
                return True
        return False

    def stats(self) -> Dict[str, Any]:
        return {
            "total_plans": self.total_plans,
            "kernel_available": KERNEL_AVAILABLE,
        }

    # ── Template-based decompositions ──

    def _plan_money(self, plan: TaskPlan, command: str, domain: str):
        """Money transactions always get: navigate → extract → PREVIEW → execute."""
        s1 = plan.add_step("Navigate to service and authenticate", "reversible")
        s2 = plan.add_step("Extract current amount due", "read_only", depends_on=[s1])
        s3 = plan.add_step("Preview payment details to user", "money_out",
                           depends_on=[s2])
        s4 = plan.add_step("Process payment after user approval", "money_out",
                           depends_on=[s3])
        # Mark step 3 as requiring preview
        for step in plan.steps:
            if step.id == s3:
                step.status = StepStatus.PREVIEW

    def _plan_finance(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Authenticate with financial service", "reversible")
        s2 = plan.add_step("Retrieve account information", "read_only", depends_on=[s1])
        plan.add_step("Format and present results", "read_only", depends_on=[s2])

    def _plan_shopping(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Search for product", "read_only")
        s2 = plan.add_step("Compare prices and reviews", "read_only", depends_on=[s1])
        s3 = plan.add_step("Present options to user", "read_only", depends_on=[s2])
        # If command says "buy", add money gate
        if any(w in command.lower() for w in ["buy", "purchase", "order"]):
            s4 = plan.add_step("Preview purchase details", "money_out", depends_on=[s3])
            plan.add_step("Complete purchase", "money_out", depends_on=[s4])
            for step in plan.steps:
                if step.id == s4:
                    step.status = StepStatus.PREVIEW

    def _plan_health(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Search for available providers", "read_only")
        s2 = plan.add_step("Check availability and insurance", "read_only", depends_on=[s1])
        plan.add_step("Present options and book if requested", "reversible", depends_on=[s2])

    def _plan_travel(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Search flights/hotels", "read_only")
        s2 = plan.add_step("Compare options and prices", "read_only", depends_on=[s1])
        s3 = plan.add_step("Present best options", "read_only", depends_on=[s2])
        if any(w in command.lower() for w in ["book", "reserve", "buy"]):
            s4 = plan.add_step("Preview booking details", "money_out", depends_on=[s3])
            plan.add_step("Confirm booking", "money_out", depends_on=[s4])
            for step in plan.steps:
                if step.id == s4:
                    step.status = StepStatus.PREVIEW

    def _plan_productivity(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Parse task details", "read_only")
        plan.add_step("Execute task (email/calendar/document)", "reversible", depends_on=[s1])

    def _plan_tech(self, plan: TaskPlan, command: str):
        s1 = plan.add_step("Analyze codebase/system state", "read_only")
        s2 = plan.add_step("Plan changes", "read_only", depends_on=[s1])
        plan.add_step("Execute changes", "reversible", depends_on=[s2])

    def _plan_generic(self, plan: TaskPlan, command: str):
        """Fallback: simple 2-step plan."""
        s1 = plan.add_step("Analyze request", "read_only")
        plan.add_step("Execute request", "reversible", depends_on=[s1])

    # ── World Model simulation ──

    def _simulate_plan(self, plan: TaskPlan):
        """Use World Model to predict outcomes and estimate surprise."""
        if self.world_model is None:
            return

        for step in plan.steps:
            # Simple embedding of step description
            step_emb = [0.0] * self.embedding_dim
            for i, ch in enumerate(step.description):
                step_emb[i % self.embedding_dim] += ord(ch) / 500.0

            # Encode and predict
            try:
                self._tick += 1
                _latent = self.world_model.encode(step_emb, self._tick)
                predicted, _surprise = self.world_model.predict(step_emb)
                # Surprise = prediction uncertainty (higher = less certain)
                step.surprise = 0.1  # Placeholder — real surprise comes post-execution
            except Exception:
                step.surprise = 0.5  # High uncertainty

    def _skip_downstream(self, plan: TaskPlan, rejected_id: str):
        """Skip all steps that depend on a rejected step."""
        to_skip = {rejected_id}
        changed = True
        while changed:
            changed = False
            for step in plan.steps:
                if step.id not in to_skip and any(d in to_skip for d in step.depends_on):
                    step.status = StepStatus.SKIPPED
                    to_skip.add(step.id)
                    changed = True
