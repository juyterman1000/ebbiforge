"""
Ebbiforge — High-Performance Multi-Agent AI Framework

The developer framework for building AI agent systems at scale.
100M agents in Rust, selective LLM reasoning, belief provenance,
enterprise compliance, and plug-and-play data connectors.

Core primitives:
    Agent, Swarm, Runtime, Task, Memory

Connector interface:
    from ebbiforge.connectors import DataSource, Signal, HTTPPoller

Output interface:
    from ebbiforge.outputs import OutputSink, SwarmEvent, ConsoleOutput

CLI:
    ebbiforge demo | benchmark | example <name> | version
"""

from ebbiforge.memory import Memory
from ebbiforge.task import Task
from ebbiforge.agent import Agent
from ebbiforge.swarm import Swarm
from ebbiforge.runtime import Runtime

# Re-export key connector/output types for convenience
from ebbiforge.connectors.base import DataSource, Signal
from ebbiforge.outputs.base import OutputSink, SwarmEvent

__version__ = "4.0.0"
__all__ = [
    "Agent", "Swarm", "Runtime", "Task", "Memory",
    "DataSource", "Signal", "OutputSink", "SwarmEvent",
]

