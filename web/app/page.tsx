'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { AgentCanvas } from '../components/swarm/AgentCanvas';
import { R0Indicator } from '../components/swarm/R0Indicator';
import { SparklineStrip } from '../components/swarm/SparklineStrip';
import { NarrationPanel } from '../components/swarm/NarrationPanel';
import { useWasmEngine } from '../hooks/useWasmEngine';
import { useSimulationLoop } from '../hooks/useSimulationLoop';
import { useTimeSeriesHistory } from '../hooks/useTimeSeriesHistory';
import { useNarration } from '../hooks/useNarration';
import { useRealDataFeed } from '../hooks/useRealDataFeed';
import { PRESETS } from '../lib/presets';
import type { ISwarmRenderer } from '../lib/types';

export default function Home() {
  const {
    ready, loadProgress, getHandle, applyPreset,
    currentAgentCount, engineRef,
  } = useWasmEngine();

  const rendererRef = useRef<ISwarmRenderer | null>(null);
  const { buffers, record, clear: clearHistory } = useTimeSeriesHistory();
  const { narrations, loading: narrationLoading, checkAndNarrate, setRealWorldData } = useNarration();

  // Real data feed: inject crypto + GitHub signals into the WASM swarm
  const injectShock = useCallback((x: number, y: number, radius: number, intensity: number) => {
    const handle = getHandle();
    if (handle) handle.engine.inject_surprise(x, y, radius, intensity);
  }, [getHandle]);
  const engineWorldSize = getHandle()?.worldSize ?? 1000;
  const realWorld = useRealDataFeed(injectShock, engineWorldSize);

  // Sync real-world data into narration context
  useEffect(() => {
    if (realWorld.crypto || realWorld.github) {
      setRealWorldData({ crypto: realWorld.crypto, github: realWorld.github });
    }
  }, [realWorld.crypto, realWorld.github, setRealWorldData]);

  const { running, metrics, colorMode, setColorMode, start, pause, toggle } = useSimulationLoop(
    getHandle,
    rendererRef,
    (m) => {
      record(m);
      checkAndNarrate(m);
    },
  );

  const [showInfo, setShowInfo] = useState(true);

  // Auto-start + apply Evolution Lab preset once WASM is ready
  const autoStartedRef = useRef(false);
  useEffect(() => {
    if (ready && !autoStartedRef.current) {
      autoStartedRef.current = true;
      // Enable evolution — real Darwinian dynamics
      const evoPreset = PRESETS.find(p => p.key === 'evolution');
      if (evoPreset) applyPreset(evoPreset);
      setTimeout(() => start(), 100);
    }
  }, [ready, start, applyPreset]);

  const handleRendererReady = useCallback((renderer: ISwarmRenderer) => {
    rendererRef.current = renderer;
  }, []);

  // Derived values — all from the real engine, zero hardcoding
  const surprisedPct = metrics.nAgents > 0 ? (metrics.surprisedCount / metrics.nAgents) * 100 : 0;
  const statusColor = surprisedPct > 8 ? '#f87171' : surprisedPct > 4 ? '#fbbf24' : '#4ade80';

  // Loading screen
  if (!ready) {
    return (
      <main className="fixed inset-0 bg-[#050812] flex items-center justify-center">
        <div className="text-center">
          <div className="relative w-24 h-24 mx-auto mb-8">
            <div className="absolute inset-0 rounded-full border-2 border-emerald-500/20 animate-ping" />
            <div className="absolute inset-2 rounded-full border-2 border-cyan-500/30 animate-ping" style={{ animationDelay: '0.3s' }} />
            <div className="absolute inset-4 rounded-full border-2 border-purple-500/40 animate-ping" style={{ animationDelay: '0.6s' }} />
            <div className="absolute inset-0 flex items-center justify-center text-4xl">🧬</div>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent mb-3">
            OpenRustSwarm
          </h1>
          <p className="text-sm text-slate-400 mb-6 max-w-sm mx-auto">
            {loadProgress || 'Compiling Rust → WebAssembly...'}
          </p>
          <div className="w-64 h-1 bg-white/5 rounded-full mx-auto overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-emerald-500 via-cyan-500 to-purple-500"
              style={{
                width: '70%',
                animation: 'loading 2s ease-in-out infinite',
              }}
            />
          </div>
          <p className="text-[0.6rem] text-slate-600 font-mono mt-6">
            Initializing {currentAgentCount.toLocaleString()} agents with 6 heritable genes each
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="fixed inset-0 bg-[#050812] text-white overflow-hidden">
      {/* ── FULLSCREEN LIVE SWARM ── */}
      <AgentCanvas
        getHandle={getHandle}
        activeTool="shock"
        agentCount={currentAgentCount}
        onRendererReady={handleRendererReady}
      />

      {/* ── R₀ TENSION OVERLAY — makes the whole canvas feel dangerous when R₀ > 1.0 ── */}
      {metrics.r0Eff > 0.8 && (
        <div
          className="absolute inset-0 pointer-events-none z-10 transition-opacity duration-1000"
          style={{
            background: metrics.r0Eff > 1.2
              ? `radial-gradient(ellipse at center, transparent 30%, rgba(239, 68, 68, ${Math.min((metrics.r0Eff - 1.0) * 0.25, 0.3)}) 100%)`
              : metrics.r0Eff > 1.0
              ? `radial-gradient(ellipse at center, transparent 40%, rgba(251, 191, 36, ${Math.min((metrics.r0Eff - 0.8) * 0.15, 0.15)}) 100%)`
              : `radial-gradient(ellipse at center, transparent 50%, rgba(251, 191, 36, ${Math.min((metrics.r0Eff - 0.8) * 0.08, 0.06)}) 100%)`,
            animation: metrics.r0Eff > 1.2 ? 'tensionPulse 2s ease-in-out infinite' : undefined,
          }}
        />
      )}

      {/* Tension pulse animation */}
      <style jsx>{`
        @keyframes tensionPulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.6; }
        }
      `}</style>
      <div className="absolute top-5 left-5 z-30 pointer-events-none">
        <div className="pointer-events-auto">
          <R0Indicator r0Eff={metrics.r0Eff} r0Base={metrics.r0Base} />
        </div>
        <div className="flex items-center gap-3 mt-2 text-[0.6rem] font-mono text-slate-400">
          <span>{metrics.nAgents.toLocaleString()} agents</span>
          <span className="text-slate-600">·</span>
          <span>tick {metrics.tick.toLocaleString()}</span>
          <span className="text-slate-600">·</span>
          <span>gen {metrics.meanGeneration.toFixed(1)}</span>
        </div>
        <div className="text-[0.5rem] font-mono text-slate-600 mt-0.5">
          {metrics.fps} FPS · {metrics.tickMs.toFixed(1)}ms/tick · diversity σ={metrics.geneDiversity.toFixed(4)}
        </div>
      </div>

      {/* ── TOP-RIGHT: Live Stats Strip ── */}
      <div className="absolute top-5 right-5 z-30 pointer-events-auto">
        <div className="flex gap-2">
          <StatPill label="Surprise" value={metrics.meanSurprise.toFixed(4)} color={statusColor} />
          <StatPill label="Health" value={metrics.meanHealth.toFixed(4)} color="#4ade80" />
          <StatPill label="Refractory" value={metrics.meanRefractory.toFixed(4)} color="#60a5fa" />
          <StatPill
            label="Surprised"
            value={`${metrics.surprisedCount.toLocaleString()} (${surprisedPct.toFixed(1)}%)`}
            color={surprisedPct > 5 ? '#f87171' : '#94a3b8'}
          />
        </div>
        {/* ── Live Data Feed Ticker ── */}
        {realWorld.crypto && (
          <div className="flex gap-2 mt-2">
            <CryptoTicker symbol="BTC" price={realWorld.crypto.bitcoin.usd} change={realWorld.crypto.bitcoin.usd_24h_change} />
            <CryptoTicker symbol="ETH" price={realWorld.crypto.ethereum.usd} change={realWorld.crypto.ethereum.usd_24h_change} />
            <CryptoTicker symbol="SOL" price={realWorld.crypto.solana.usd} change={realWorld.crypto.solana.usd_24h_change} />
            {realWorld.github && (
              <div className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-sm border border-white/5">
                <div className="text-[0.45rem] uppercase tracking-widest text-slate-600 mb-0.5">GitHub</div>
                <div className="text-[0.65rem] font-mono font-bold text-purple-400">
                  {realWorld.github.summary.totalEvents} events
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── CENTER: Hero Text (dismissable) ── */}
      {showInfo && (
        <div className="absolute inset-0 z-20 flex items-center justify-center pointer-events-none">
          <div
            className="pointer-events-auto max-w-lg text-center p-8 rounded-3xl backdrop-blur-xl bg-black/50 border border-white/10 shadow-[0_0_80px_rgba(0,0,0,0.5)] cursor-pointer"
            onClick={() => setShowInfo(false)}
          >
            <h1 className="text-4xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-300 to-purple-400 bg-clip-text text-transparent mb-3 tracking-tight">
              OpenRustSwarm
            </h1>
            <p className="text-lg text-slate-300 mb-4 leading-relaxed">
              {metrics.nAgents.toLocaleString()} agents evolving in your browser right now.
            </p>
            <p className="text-sm text-slate-400 mb-5 leading-relaxed">
              Real SIRS epidemiology. Real Darwinian evolution with {metrics.evolutionEnabled ? '6' : '0'} heritable genes.
              6-channel stigmergic pheromone field. Spatial hash grid with Fibonacci hashing.
              All running in Rust → WebAssembly at {metrics.fps} FPS.
            </p>

            <div className="flex items-center justify-center gap-6 mb-5 text-xs text-slate-500 font-mono">
              <span>R₀ = {metrics.r0Eff.toFixed(3)}</span>
              <span>·</span>
              <span>Gen {metrics.meanGeneration.toFixed(1)}</span>
              <span>·</span>
              <span>{metrics.surprisedCount.toLocaleString()} surprised</span>
            </div>

            <div className="flex items-center justify-center gap-3">
              <button
                onClick={(e) => { e.stopPropagation(); setShowInfo(false); }}
                className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 text-white text-sm font-semibold hover:from-emerald-500 hover:to-cyan-500 transition-all shadow-[0_0_30px_rgba(16,185,129,0.3)]"
              >
                Click anywhere to inject surprise →
              </button>
              <Link
                href="/swarm"
                onClick={(e) => e.stopPropagation()}
                className="px-5 py-2.5 rounded-xl border border-white/10 text-slate-300 text-sm font-medium hover:bg-white/5 hover:border-white/20 transition-all"
              >
                Full Observatory
              </Link>
            </div>

            <p className="text-[0.55rem] text-slate-600 mt-4">
              Click to dismiss · Click anywhere on the swarm to inject a surprise shockwave
            </p>
          </div>
        </div>
      )}

      {/* ── BOTTOM: Real-time Sparklines ── */}
      <SparklineStrip buffers={buffers} evolutionEnabled={metrics.evolutionEnabled} />

      {/* ── BOTTOM-RIGHT: Minimal branding ── */}
      <div className="absolute bottom-14 right-4 z-30 text-right">
        <div className="flex items-center gap-2 justify-end">
          <div
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: statusColor, boxShadow: `0 0 8px ${statusColor}` }}
          />
          <span className="text-[0.55rem] font-mono text-slate-500">
            {running ? 'LIVE' : 'PAUSED'} · Rust/WASM · v4.0.0
          </span>
        </div>
      </div>

      {/* ── BOTTOM-LEFT: Live Narration Panel ── */}
      <NarrationPanel narrations={narrations} loading={narrationLoading} />

      {/* ── BOTTOM-LEFT: Keyboard hints ── */}
      <div className="absolute bottom-14 left-4 z-30">
        <span className="text-[0.5rem] font-mono text-slate-700">
          Space pause · Click to shock · {metrics.evolutionEnabled ? 'Evolution ON' : 'Evolution OFF'}
        </span>
      </div>

      {/* Keyboard handler */}
      <KeyboardHandler toggle={toggle} />
    </main>
  );
}

// ─── Tiny stat pill ──────────────────────────────────────────────────
function StatPill({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-sm border border-white/5">
      <div className="text-[0.45rem] uppercase tracking-widest text-slate-600 mb-0.5">{label}</div>
      <div className="text-[0.65rem] font-mono font-bold" style={{ color }}>{value}</div>
    </div>
  );
}

// ─── Live crypto ticker ─────────────────────────────────────────────
function CryptoTicker({ symbol, price, change }: { symbol: string; price: number; change: number }) {
  const isUp = change >= 0;
  const color = isUp ? '#4ade80' : '#f87171';
  const arrow = isUp ? '▲' : '▼';
  return (
    <div className="px-2.5 py-1.5 rounded-lg bg-black/40 backdrop-blur-sm border border-white/5">
      <div className="text-[0.45rem] uppercase tracking-widest text-slate-600 mb-0.5">{symbol}</div>
      <div className="flex items-center gap-1">
        <span className="text-[0.65rem] font-mono font-bold text-slate-200">
          ${price >= 1000 ? (price / 1000).toFixed(1) + 'k' : price.toFixed(2)}
        </span>
        <span className="text-[0.5rem] font-mono" style={{ color }}>
          {arrow}{Math.abs(change).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

// ─── Keyboard handler (space to pause) ──────────────────────────────
function KeyboardHandler({ toggle }: { toggle: () => void }) {
  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === ' ') {
        e.preventDefault();
        toggle();
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [toggle]);

  return null;
}
