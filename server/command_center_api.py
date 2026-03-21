"""
Command Center API Routes — FastAPI endpoints for the Command Center UI.

Bridges:
  Next.js Command Center UI  ←→  Python Orchestrator  ←→  Rust Kernel

Endpoints:
  POST /api/command-center/command   →  Process user command
  POST /api/command-center/approve   →  Approve money gate step
  GET  /api/command-center/stats     →  Get orchestrator stats
"""

from __future__ import annotations

import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from orchestrator import Orchestrator
from llm_gateway import LlmGateway

# ── Singleton instances ──
_orchestrator: Optional[Orchestrator] = None
_gateway: Optional[LlmGateway] = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        gateway = get_gateway()
        _orchestrator = Orchestrator(
            embedding_dim=64,
            llm_fn=lambda prompt: gateway.call(prompt).text,
        )
    return _orchestrator


def get_gateway() -> LlmGateway:
    global _gateway
    if _gateway is None:
        _gateway = LlmGateway(max_cost_per_session=1.0)
    return _gateway


# ── FastAPI Router ──
router = APIRouter(prefix="/api/command-center")


class CommandRequest(BaseModel):
    command: str


class ApproveRequest(BaseModel):
    agent_id: str
    step_id: str


@router.post("/command")
def process_command(req: CommandRequest):
    """Process a user command through the full pipeline."""
    orch = get_orchestrator()
    return orch.process_command(req.command)


@router.post("/approve")
def approve_step(req: ApproveRequest):
    """Approve a money gate step."""
    orch = get_orchestrator()
    return orch.approve_step(req.agent_id, req.step_id)


@router.post("/reject")
def reject_step(req: ApproveRequest):
    """Reject a money gate step."""
    orch = get_orchestrator()
    return orch.reject_step(req.agent_id, req.step_id)


@router.get("/stats")
def get_stats():
    """Get orchestrator + gateway stats."""
    orch = get_orchestrator()
    gw = get_gateway()
    return {
        **orch.stats(),
        "gateway": gw.stats(),
    }
