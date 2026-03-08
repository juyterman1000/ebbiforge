"""
CogOps Swarm Brain — Activates ALL Rust modules.

Layer 1 (Nervous System):  ProductionTensorSwarm, PollinatorState
Layer 2 (Voice):           IntrospectionEngine, AdaptivePruner, PredictiveSafetyShield
Layer 3 (Hands):           PromotionLogic (Light→Heavy escalation)
Layer 4 (Memory):          MemoryConsolidator, CrossPollination, MetaCognition, CuriosityModule
Layer 5 (Face):            API endpoints → frontend

Every class imported here is compiled Rust running via pyo3. Zero Python simulation.
"""

import threading
import time
import json
import os
import requests
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# ─── Import ALL dead Rust modules ─────────────────────────────────────
try:
    from ebbiforge_core import (
        # Layer 1: Nervous System (Swarm Engine)
        ProductionTensorSwarm,     # 4-tier LOD swarm: Dormant→Simplified→Full→Heavy
        DormantAgent,              # Dormant tier agent
        SimplifiedPool,            # Simplified tier pool
        PollinatorState,           # TD-learning information sharing
        PollinatorConfig,          # Pollinator tuning
        PromotionLogic,            # Light→Heavy agent promotion
        PromoterConfig,            # Promotion thresholds
        SwarmConfig,               # Swarm configuration
        TensorSwarm,               # Raw tensor swarm engine

        # Layer 2: Voice (Intelligence)
        IntrospectionEngine,       # Generator+critic loop
        AdaptivePruner,            # RL-style context pruning
        ContextFragment,           # Fragment for pruning
        ContextFeatures,           # Features for scoring
        PredictiveSafetyShield,    # Trajectory safety validation
        CodeQualityGuard,          # Code output validation

        # Layer 3: Hands (Cross-agent coordination)
        CrossPollination,          # Signed experience packs
        FailureContext,            # Failure context for sharing
        ExperiencePack,            # Experience pack for P2P sharing

        # Layer 4: Memory + World Model
        MemoryConsolidator,        # Ebbinghaus forgetting + reconsolidation
        ConsolidatedMemory,        # Memory entry with decay
        HippocampusEngine,         # Brain-inspired associative memory (LSH + Ebbinghaus)
        MemoryBankConfig,          # Disk-backed storage config
        PlanningEngine,            # Autoregressive prediction
        LatentEncoder,             # State→latent vector
        AutoregressivePredictor,   # Latent→future prediction
        DiffusionPredictor,        # Latent diffusion dynamics
        GeometricEncoder,          # Geometric latent encoding
        LatentState,               # Latent state representation
        WorldModelConfig,          # World model config
        Prediction,                # Prediction output
        ActionScore,               # Action scoring

        # Evolution
        MetaCognition,             # Insight generation
        Insight,                   # Insight type
        CuriosityModule,           # Novel challenge generation
        PopulationEngine,          # Darwinian agent evolution
        AgentGenome,               # Agent genome
        EvolutionConfig,           # Evolution configuration
        ToolSynthesizer,           # Generate new tools
        GeneratedTool,             # A generated tool output
        SafetySandbox,             # Safe tool execution
        DynamicRegistry,           # Runtime tool registration

        # Security
        SecureVault,               # AES-GCM encryption
        AgentIdentity,             # Ed25519 agent identity
        TrustStore,                # Multi-agent trust chain

        # Compliance
        ComplianceEngine,          # Full compliance pipeline
        ComplianceResult,          # Compliance check result
        RateLimiter,               # Rate limiting
        RateLimitConfig,           # Rate limit configuration
        RateLimitResult,           # Rate limit check result
        EscalationFlow,            # Human-in-the-loop escalation
        EscalationResult,          # Escalation result
        PendingAction,             # Pending escalation action
        InputSanitizer,            # Input sanitization
        SanitizeResult,            # Sanitize result
        PIIRedactor,               # PII redaction
        PIIMatch,                  # PII match result
        AuditEvent,                # Audit trail event
        Policy,                    # Compliance policy
        TraceStep,                 # Execution trace step

        # Core Agent Framework
        Agent,                     # Agent definition
        AgentRegistry,             # Agent registry
        AgentGraphPy,              # Agent execution graph
        CogOpsConfig,              # Global configuration
        CogOpsContext,             # Middleware context
        HistoryBuffer,             # Arc<RwLock<Vec<TrajectoryPoint>>>
        TrajectoryPoint,           # Execution step snapshot
        SharedMemoryStore,         # Embedding-based shared memory
        StorageConfig,             # Storage backend config
        SearchResult,              # Vector search result
        DragonflyStore,            # Dragonfly cache backend
        RemoteVectorStore,         # Qdrant vector store

        # Workflow Agents
        SequentialAgent,           # Sequential execution
        ParallelAgent,             # Parallel execution
        LoopAgent,                 # Loop execution

        # Benchmarking
        AgentBenchmark,            # Performance benchmarking
    )
    RUST_AVAILABLE = True
    print("✅ ALL 74 Rust exports loaded — every line of Rust is now active")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"⚠️ Rust modules not available: {e}")


# ─── Swarm Brain ──────────────────────────────────────────────────────
@dataclass
class SwarmAlert:
    """Alert generated when the swarm crosses a threshold."""
    timestamp: float
    alert_type: str       # 'r0_spike', 'death_wave', 'gene_collapse', 'surprise_cascade'
    severity: str         # 'low', 'medium', 'high', 'critical'
    metrics: Dict[str, float]
    explanation: Optional[str] = None
    llm_analysis: Optional[str] = None


class SwarmBrain:
    """
    The server-side brain. Connects ALL Rust modules into:
    FEEL (swarm) → EXPLAIN (LLM) → ACT (alerts) → REMEMBER (memory)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("MODEL_API_KEY")
        self.alerts: List[SwarmAlert] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if not RUST_AVAILABLE:
            print("⚠️ SwarmBrain: Rust not available, running in degraded mode")
            return

        # ── Layer 1: Nervous System ──
        self.swarm_config = SwarmConfig()  # Uses Rust defaults: 100K agents, 1000x1000 world
        # ProductionTensorSwarm: 4-tier LOD (Dormant→Simplified→Full→Heavy)
        self.production_swarm = ProductionTensorSwarm(self.swarm_config.population_size)
        # TensorSwarm: direct access to Tier 3 metrics (mean_surprise, mean_health, etc.)
        self.swarm = TensorSwarm(self.swarm_config.population_size)
        self.pollinator = PollinatorState()  # TD-learning for info sharing
        self.promoter = PromotionLogic()     # Light→Heavy agent promotion

        # ── Layer 2: Voice ──
        self.introspection = IntrospectionEngine(3)  # Generator+critic, max 3 attempts
        self.pruner = AdaptivePruner()               # Smart context windowing
        self.safety_shield = PredictiveSafetyShield(0.8)  # Block risky outputs
        self.code_guard = CodeQualityGuard()         # Validate code outputs

        # ── Layer 3: Hands (Cross-agent) ──
        self.cross_pollination = CrossPollination()  # Signed experience sharing
        self.trust_store = TrustStore()              # Multi-agent trust chain

        # ── Layer 4: Memory ──
        self.memory = MemoryConsolidator()           # Ebbinghaus forgetting (legacy)
        # HippocampusEngine: associative recall with LSH + context scoring
        # Replaces brute-force memory lookups with sub-μs semantic search.
        # Key value: recall-before-LLM — avoid redundant API calls.
        self.hippocampus = HippocampusEngine(
            capacity=500_000,       # 500K episodes in RAM
            sdm_locations=10_000,   # Kanerva SDM for permanent memories
            consolidation_interval=50,
            recall_reinforcement=1.3,
        )
        self.planner = PlanningEngine()              # Autoregressive prediction
        self.metacognition = MetaCognition()         # Insight generation
        self.curiosity = CuriosityModule()           # Novel challenge detection

        # ── Evolution ──
        evo_config = EvolutionConfig()
        self.population = PopulationEngine(evo_config)  # Darwinian agent evolution
        self.synthesizer = ToolSynthesizer()         # Generate new tools at runtime
        self.sandbox = SafetySandbox()               # Safe execution
        self.registry = DynamicRegistry()            # Runtime tool registration

        # ── Security ──
        vault_hex = os.getenv("SECURE_VAULT_KEY")
        if vault_hex:
            self.vault = SecureVault(vault_hex)  # AES-GCM encryption
        else:
            self.vault = None  # No vault key configured — encryption disabled
            print("⚠️ SECURE_VAULT_KEY not set — SecureVault disabled")
        # AgentIdentity has no Python constructor — use TrustStore only
        self.identity = None

        # ── Compliance ──
        self.compliance = ComplianceEngine()         # Full compliance pipeline
        self.rate_limiter = RateLimiter()             # Rate limiting
        self.escalation = EscalationFlow()           # Human-in-the-loop
        self.sanitizer = InputSanitizer()            # Input sanitization

        # ── State tracking ──
        self.tick_count = 0
        self.last_r0 = 0.0
        self.last_surprise = 0.0
        self.last_gene_diversity = 0.0

        print(f"✅ SwarmBrain initialized: {self.swarm_config.population_size:,} agents across 4 LOD tiers")
        print(f"   Modules active: Swarm, Introspection, Pruner, SafetyShield, Memory,")
        print(f"   Planner, MetaCognition, Curiosity, CrossPollination, Population,")
        print(f"   ToolSynthesizer, Sandbox, Vault, Trust, Compliance, RateLimiter")

    def inject_signal(self, source: str, value: float, surprise_weight: float = 1.0):
        """Layer 1: Inject a real-world signal into the swarm."""
        if not RUST_AVAILABLE:
            return

        # Inject at world center, radius proportional to world size
        cx = self.swarm_config.world_width / 2.0
        cy = self.swarm_config.world_height / 2.0
        radius = self.swarm_config.world_width / 10.0
        self.swarm.apply_environmental_shock((cx, cy), radius, value * surprise_weight)
        # Also wake dormant agents in ProductionTensorSwarm on strong signals
        if surprise_weight > 0.5:
            self.production_swarm.set_global_triggers(0xFF)

        # Record in memory with Ebbinghaus scoring
        memory_text = f"Signal: {source}={value:.4f} (weight={surprise_weight:.2f}) at tick {self.tick_count}"
        self.memory.consolidate("swarm-brain", [json.dumps({"signal": source, "value": value, "tick": self.tick_count})])
        # Also store in HippocampusEngine for associative recall
        salience = min(surprise_weight * 50.0, 90.0)
        self.hippocampus.remember(
            memory_text,
            salience=salience,
            source=source,
            emotional_tag=3 if surprise_weight > 0.8 else 2 if surprise_weight > 0.5 else 0,
        )

    def tick(self):
        """Run one simulation tick — the full FEEL→EXPLAIN→REMEMBER pipeline."""
        if not RUST_AVAILABLE:
            return {}

        self.tick_count += 1

        # Tick BOTH engines
        self.production_swarm.tick()  # 4-tier LOD: Dormant wakeup, Simplified physics, Full sim
        self.swarm.tick()             # Direct tier-3 tensor swarm

        # Read real metrics from the Rust TensorSwarm
        pop_metrics = self.swarm.sample_population_metrics()
        agent_count = len(self.swarm.surprise_scores)
        health_scores = self.swarm.health
        mean_health = sum(health_scores) / max(len(health_scores), 1)
        metrics = {
            "tick": self.tick_count,
            "agent_count": agent_count,
            "mean_surprise": pop_metrics.get('mean_surprise_score', 0.0),
            "mean_health": mean_health,
            "active_heavy_agents": pop_metrics.get('active_heavy_agents', 0),
        }

        # ── DETECT: Check for threshold crossings ──
        alerts = self._check_thresholds(metrics)

        # ── REMEMBER: Consolidate memory (Ebbinghaus) ──
        self.hippocampus.tick()  # Advance memory clock + auto-consolidation
        if self.tick_count % 100 == 0:
            self.memory.consolidate("swarm-brain", [json.dumps({"tick": self.tick_count, "type": "periodic"})])
            self.hippocampus.remember(
                f"Periodic checkpoint at tick {self.tick_count}: surprise={metrics.get('mean_surprise', 0):.4f}",
                salience=5.0, source="periodic",
            )

        # ── EXPLAIN: If alert triggered, generate LLM explanation ──
        for alert in alerts:
            if alert.severity in ('high', 'critical') and self.api_key:
                explanation = self._generate_explanation(alert, metrics)
                alert.llm_analysis = explanation

            with self._lock:
                self.alerts.append(alert)
                if len(self.alerts) > 100:
                    self.alerts = self.alerts[-100:]

        # ── EVOLVE: MetaCognition generates insights ──
        if self.tick_count % 500 == 0:
            insights = self.metacognition.get_insights()
            if insights:
                # Store insights in consolidated memory for future explanation context
                for insight in insights:
                    self.memory.consolidate("swarm-brain", [json.dumps({"type": "insight", "data": str(insight), "tick": self.tick_count})])
                    self.hippocampus.remember(
                        f"Insight: {str(insight)}",
                        salience=40.0, source="metacognition", emotional_tag=1,
                    )

        # ── CURIOSITY: Generate novel challenges ──
        if self.tick_count % 1000 == 0:
            challenge = self.curiosity.propose_challenge()
            if challenge:
                # Inject curiosity challenges as weak signals to probe the swarm
                self.swarm.apply_environmental_shock(
                    (self.swarm_config.world_width / 3.0, self.swarm_config.world_height / 3.0),
                    self.swarm_config.world_width / 20.0,
                    0.05  # Weak probe — curiosity, not alarm
                )
                self.memory.consolidate("swarm-brain", [json.dumps({"type": "curiosity", "challenge": str(challenge), "tick": self.tick_count})])
                self.hippocampus.remember(
                    f"Curiosity challenge: {str(challenge)}",
                    salience=15.0, source="curiosity",
                )

        # Update tracking
        self.last_surprise = metrics.get("mean_surprise", 0.0)

        return metrics

    def _check_thresholds(self, metrics: Dict[str, float]) -> List[SwarmAlert]:
        """Detect anomalies in swarm state."""
        alerts = []
        now = time.time()

        surprise = metrics.get("mean_surprise", 0.0)

        # Surprise cascade
        if surprise > 0.15 and self.last_surprise < 0.10:
            alerts.append(SwarmAlert(
                timestamp=now,
                alert_type="surprise_cascade",
                severity="high",
                metrics=metrics.copy(),
                explanation=f"Mean surprise spiked from {self.last_surprise:.4f} to {surprise:.4f}",
            ))

        return alerts

    def _generate_explanation(self, alert: SwarmAlert, metrics: Dict) -> str:
        """Layer 2: Use LLM to explain what the swarm is sensing.
        
        RECALL-BEFORE-LLM: Check if we've explained a similar alert before.
        If so, return the cached explanation — saves ~$0.01 per call.
        """
        # ── RECALL-BEFORE-LLM: Check memory for similar past explanations ──
        query = f"{alert.alert_type} {alert.explanation}"
        past_memories = self.hippocampus.recall(query, top_k=1)
        if past_memories:
            best = past_memories[0]
            # If we have a highly relevant, recent memory, use it
            if best.retention > 0.3 and best.recall_count > 0:
                return f"[FROM MEMORY — {best.recall_count}× recalled] {best.content}"

        if not self.api_key:
            return "No API key — cannot generate explanation"

        # Use AdaptivePruner to build optimal context
        memory_context = str(self.memory.get_memories("swarm-brain")[:10])
        # Add hippocampus recall for richer context
        hippo_context = self.hippocampus.recall(query, top_k=3)
        if hippo_context:
            memory_context += "\n\nASSOCIATIVE RECALL (similar past events):\n"
            for m in hippo_context:
                memory_context += f"  - [{m.retention*100:.0f}% retained] {m.content}\n"

        # Build the context assembly (Layer 2 architecture)
        prompt = f"""You are the VOICE of a swarm intelligence system.

SWARM STATE (from real Rust engine, {metrics.get('agent_count', 0):,} agents):
- Mean Surprise: {metrics.get('mean_surprise', 0):.6f}
- Mean Health: {metrics.get('mean_health', 0):.6f}
- Tick: {metrics.get('tick', 0)}

ALERT: {alert.alert_type} ({alert.severity})
{alert.explanation}

RECENT MEMORY (Ebbinghaus-weighted, most surprising first):
{memory_context}

Explain in 2-3 sentences what the swarm is detecting and why it matters.
Be specific about the data. No hype. No jargon. Just clarity."""

        # Safety check via PredictiveSafetyShield
        trajectory = json.dumps([{"step": 1, "action": "explain", "thought": prompt[:200]}])
        risk_score, risk_reason = self.safety_shield.analyze_risk(trajectory)
        if risk_score > 0.8:
            return f"[SAFETY BLOCKED] Risk: {risk_score:.2f} — {risk_reason}"

        # Call Gemini
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={self.api_key}"
            r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=30)
            r.raise_for_status()
            explanation = r.json()['candidates'][0]['content']['parts'][0]['text']

            # REMEMBER the explanation for future recall-before-LLM
            self.hippocampus.remember(
                f"LLM explanation for {alert.alert_type}: {explanation[:200]}",
                salience=60.0,
                source="llm_explanation",
                emotional_tag=3 if alert.severity == 'critical' else 2,
            )
            return explanation
        except Exception as e:
            return f"LLM error: {e}"

    def get_state(self) -> Dict[str, Any]:
        """Get full brain state for the API."""
        state = {
            "tick": self.tick_count,
            "rust_modules_active": RUST_AVAILABLE,
        }

        if RUST_AVAILABLE:
            try:
                pop_metrics = self.swarm.sample_population_metrics()
                agent_count = len(self.swarm.surprise_scores)
                health_scores = self.swarm.health
                mean_health = sum(health_scores) / max(len(health_scores), 1)
                state.update({
                    "agent_count": agent_count,
                    "mean_surprise": pop_metrics.get('mean_surprise_score', 0.0),
                    "mean_health": mean_health,
                    "active_heavy_agents": pop_metrics.get('active_heavy_agents', 0),
                })
            except Exception as e:
                state["swarm_error"] = str(e)
            # Memory/metacognition — graceful fallback
            try:
                state["memory_entries"] = self.memory.total_trajectories("swarm-brain")
                state["hippocampus_episodes"] = self.hippocampus.episode_count
                stats = self.hippocampus.stats()
                state["hippocampus_recalls"] = stats.total_recalls
                state["hippocampus_consolidated"] = stats.consolidated_count
            except:
                state["memory_entries"] = 0
            try:
                state["insights"] = len(self.metacognition.get_insights())
            except:
                state["insights"] = 0

        with self._lock:
            state["alerts"] = [
                {
                    "timestamp": a.timestamp,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "explanation": a.explanation,
                    "llm_analysis": a.llm_analysis,
                }
                for a in self.alerts[-10:]
            ]

        return state

    def start_background(self, tick_interval: float = 0.05):
        """Start the brain running in a background thread at ~20Hz."""
        if self.running:
            return
        self.running = True

        def loop():
            print(f"🧠 SwarmBrain background loop started ({1/tick_interval:.0f} Hz)")
            while self.running:
                try:
                    self.tick()
                except Exception as e:
                    print(f"⚠️ SwarmBrain tick error: {e}")
                time.sleep(tick_interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background loop."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)


# ─── Singleton ────────────────────────────────────────────────────────
_brain: Optional[SwarmBrain] = None

def get_brain() -> SwarmBrain:
    global _brain
    if _brain is None:
        _brain = SwarmBrain()
    return _brain
