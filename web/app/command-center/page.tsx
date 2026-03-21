'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';

// ════════════════════════════════════════════════════════════════════
//  TYPES
// ════════════════════════════════════════════════════════════════════

interface StepResult {
  id: string;
  description: string;
  status: string;
  action_class: string;
  surprise: number;
  result?: string;
  amount?: number;
  recipient?: string;
}

interface PlanResult {
  plan_id: string;
  command: string;
  persona_id: string;
  steps: StepResult[];
  has_money_gate: boolean;
  is_complete: boolean;
}

interface IntentInfo {
  domain: string;
  vagueness: number;
  persona_id: string;
  persona_is_new: boolean;
  persona_affinity: number;
  action_class: string;
  needs_llm: boolean;
}

interface CommandResult {
  agent_id: string;
  command: string;
  intent: IntentInfo;
  lod_tier: string;
  plan: PlanResult;
  needs_approval: boolean;
  blocked_steps: StepResult[];
  is_complete: boolean;
}

// ════════════════════════════════════════════════════════════════════
//  COMMAND CENTER PAGE
// ════════════════════════════════════════════════════════════════════

export default function CommandCenter() {
  const [command, setCommand] = useState('');
  const [results, setResults] = useState<CommandResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState<Record<string, any> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input on mount
  useEffect(() => { inputRef.current?.focus(); }, []);

  // Poll stats
  useEffect(() => {
    const poll = setInterval(async () => {
      try {
        const r = await fetch('/api/command-center/stats');
        if (r.ok) setStats(await r.json());
      } catch {}
    }, 3000);
    return () => clearInterval(poll);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!command.trim() || loading) return;
    setLoading(true);
    try {
      const r = await fetch('/api/command-center/command', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: command.trim() }),
      });
      if (r.ok) {
        const data = await r.json();
        setResults(prev => [data, ...prev]);
        setCommand('');
      }
    } catch (e) {
      console.error('Command failed:', e);
    }
    setLoading(false);
  }, [command, loading]);

  const handleApprove = useCallback(async (agentId: string, stepId: string) => {
    try {
      const r = await fetch('/api/command-center/approve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId, step_id: stepId }),
      });
      if (r.ok) {
        const data = await r.json();
        setResults(prev => prev.map(res =>
          res.agent_id === agentId ? { ...res, ...data } : res
        ));
      }
    } catch (e) { console.error('Approve failed:', e); }
  }, []);

  return (
    <main className="min-h-screen bg-[#050812] text-white">
      {/* ── Header ── */}
      <header className="border-b border-white/5 bg-black/30 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-cyan-500 flex items-center justify-center text-lg">
              ⚡
            </div>
            <div>
              <h1 className="text-lg font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                Agent OS Command Center
              </h1>
              <p className="text-[0.6rem] text-slate-500 font-mono">
                {stats ? `${stats.total_tasks ?? 0} tasks · ${stats.math_ratio ? (stats.math_ratio * 100).toFixed(0) : 0}% math-only` : 'Connecting...'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {stats && (
              <div className="flex gap-2">
                <MiniStat label="LLM Calls" value={stats.total_llm_calls ?? 0} />
                <MiniStat label="Math Steps" value={stats.total_math_steps ?? 0} />
                <MiniStat label="Active" value={stats.active_agents ?? 0} color="emerald" />
              </div>
            )}
            <Link href="/" className="text-xs text-slate-500 hover:text-slate-300 transition-colors">
              ← Swarm View
            </Link>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* ── Command Input ── */}
        <div className="mb-8">
          <div className="relative">
            <input
              ref={inputRef}
              type="text"
              value={command}
              onChange={e => setCommand(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSubmit()}
              placeholder="Give a command... (e.g., 'Pay my electric bill', 'Check flight prices to NYC')"
              className="w-full px-6 py-4 rounded-2xl bg-white/[0.03] border border-white/10 text-white placeholder-slate-500 text-lg focus:outline-none focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/20 transition-all font-light"
              id="command-input"
            />
            <button
              onClick={handleSubmit}
              disabled={loading || !command.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-cyan-600 text-white text-sm font-semibold hover:from-emerald-500 hover:to-cyan-500 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-[0_0_20px_rgba(16,185,129,0.2)]"
              id="submit-command"
            >
              {loading ? '...' : 'Execute →'}
            </button>
          </div>
          <p className="text-[0.6rem] text-slate-600 mt-2 font-mono pl-2">
            Stage 1: TF-IDF domain scoring · Stage 2: Vagueness analysis · Stage 3: PSM persona routing · Stage 4: Plan decomposition
          </p>
        </div>

        {/* ── Results ── */}
        <div className="space-y-6">
          {results.map((result, i) => (
            <ResultCard
              key={`${result.agent_id}-${i}`}
              result={result}
              onApprove={handleApprove}
            />
          ))}
        </div>

        {/* Empty state */}
        {results.length === 0 && (
          <div className="text-center py-24">
            <div className="text-6xl mb-6 opacity-30">🤖</div>
            <h2 className="text-xl font-light text-slate-400 mb-2">Your Agent Army Awaits</h2>
            <p className="text-sm text-slate-600 max-w-md mx-auto">
              Type a command above. Personas emerge from YOUR task patterns — born, evolve,
              reproduce, merge, split, die. Each task is routed through pure Rust math
              before any LLM is called.
            </p>
            <div className="flex justify-center gap-3 mt-8">
              {['Pay my electric bill', 'Book a flight to NYC', 'Check my calendar'].map(ex => (
                <button
                  key={ex}
                  onClick={() => setCommand(ex)}
                  className="px-4 py-2 rounded-xl border border-white/5 text-xs text-slate-500 hover:text-slate-300 hover:border-white/10 transition-all"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

// ════════════════════════════════════════════════════════════════════
//  RESULT CARD — Shows the full pipeline result
// ════════════════════════════════════════════════════════════════════

function ResultCard({ result, onApprove }: {
  result: CommandResult;
  onApprove: (agentId: string, stepId: string) => void;
}) {
  const intent = result.intent;

  return (
    <div className="rounded-2xl border border-white/5 bg-white/[0.02] overflow-hidden" id={`result-${result.agent_id}`}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <DomainIcon domain={intent.domain} />
          <div>
            <p className="text-sm font-medium text-slate-200">{result.command}</p>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-[0.6rem] font-mono text-slate-500">
                {intent.domain}
              </span>
              <ClassBadge actionClass={intent.action_class} />
              <span className="text-[0.6rem] font-mono text-slate-600">
                LOD: {result.lod_tier}
              </span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center gap-2">
            {intent.persona_is_new && (
              <span className="px-2 py-0.5 rounded-md bg-purple-500/10 text-purple-400 text-[0.6rem] font-mono border border-purple-500/20">
                🎂 NEW PERSONA
              </span>
            )}
            <span className="text-[0.65rem] font-mono text-cyan-400">
              {intent.persona_id}
            </span>
          </div>
          <p className="text-[0.55rem] text-slate-600 font-mono mt-0.5">
            affinity: {intent.persona_affinity.toFixed(3)} · vagueness: {intent.vagueness.toFixed(2)}
            {intent.needs_llm ? ' · LLM needed' : ' · math-only'}
          </p>
        </div>
      </div>

      {/* Task DAG */}
      <div className="px-6 py-4">
        <p className="text-[0.55rem] uppercase tracking-widest text-slate-600 mb-3">
          Task Plan · {result.plan.steps.length} steps
        </p>
        <div className="space-y-2">
          {result.plan.steps.map((step, i) => (
            <StepRow
              key={step.id}
              step={step}
              index={i}
              isLast={i === result.plan.steps.length - 1}
              onApprove={() => onApprove(result.agent_id, step.id)}
            />
          ))}
        </div>
      </div>

      {/* Money Gate Preview */}
      {result.needs_approval && result.blocked_steps.length > 0 && (
        <div className="px-6 py-4 bg-amber-500/5 border-t border-amber-500/10">
          <div className="flex items-center gap-2 mb-3">
            <span className="text-amber-400 text-lg">⚡</span>
            <span className="text-sm font-semibold text-amber-300">Payment Preview Required</span>
          </div>
          {result.blocked_steps.map(step => (
            <MoneyGatePreview
              key={step.id}
              step={step}
              onApprove={() => onApprove(result.agent_id, step.id)}
            />
          ))}
        </div>
      )}

      {/* Status bar */}
      <div className="px-6 py-2 bg-white/[0.01] border-t border-white/5 flex items-center justify-between">
        <span className="text-[0.55rem] font-mono text-slate-600">
          agent: {result.agent_id}
        </span>
        <span className={`text-[0.55rem] font-mono ${result.is_complete ? 'text-emerald-400' : 'text-amber-400'}`}>
          {result.is_complete ? '✓ Complete' : '⏳ In Progress'}
        </span>
      </div>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
//  SUB-COMPONENTS
// ════════════════════════════════════════════════════════════════════

function StepRow({ step, index, isLast, onApprove }: {
  step: StepResult; index: number; isLast: boolean;
  onApprove: () => void;
}) {
  const statusColors: Record<string, string> = {
    completed: 'bg-emerald-500',
    running: 'bg-cyan-500 animate-pulse',
    pending: 'bg-slate-600',
    preview: 'bg-amber-500 animate-pulse',
    blocked: 'bg-red-500',
    failed: 'bg-red-500',
    skipped: 'bg-slate-700',
  };

  return (
    <div className="flex items-start gap-3">
      {/* Timeline connector */}
      <div className="flex flex-col items-center mt-1">
        <div className={`w-2.5 h-2.5 rounded-full ${statusColors[step.status] ?? 'bg-slate-600'}`} />
        {!isLast && <div className="w-px h-6 bg-white/5 mt-1" />}
      </div>
      {/* Step content */}
      <div className="flex-1 pb-2">
        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-300">{step.description}</p>
          <div className="flex items-center gap-2">
            <ClassBadge actionClass={step.action_class} small />
            {step.status === 'preview' && (
              <button
                onClick={onApprove}
                className="px-2.5 py-1 rounded-lg bg-amber-500/10 text-amber-400 text-[0.6rem] font-semibold border border-amber-500/20 hover:bg-amber-500/20 transition-all"
              >
                Approve ✓
              </button>
            )}
          </div>
        </div>
        {step.result && (
          <p className="text-[0.6rem] text-slate-500 mt-0.5 font-mono">{step.result.slice(0, 120)}</p>
        )}
      </div>
    </div>
  );
}

function MoneyGatePreview({ step, onApprove }: {
  step: StepResult;
  onApprove: () => void;
}) {
  return (
    <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 flex items-center justify-between">
      <div>
        <p className="text-sm text-amber-200 font-medium">{step.description}</p>
        {step.amount && (
          <p className="text-lg font-bold text-amber-100 mt-1">
            ${step.amount.toFixed(2)}
            {step.recipient && <span className="text-sm font-normal text-amber-300 ml-2">→ {step.recipient}</span>}
          </p>
        )}
      </div>
      <div className="flex gap-2">
        <button
          onClick={onApprove}
          className="px-4 py-2 rounded-xl bg-emerald-600 text-white text-sm font-semibold hover:bg-emerald-500 transition-all shadow-[0_0_20px_rgba(16,185,129,0.2)]"
          id={`approve-${step.id}`}
        >
          ✅ Approve
        </button>
        <button className="px-4 py-2 rounded-xl border border-white/10 text-slate-400 text-sm hover:bg-white/5 transition-all">
          ❌ Cancel
        </button>
      </div>
    </div>
  );
}

function DomainIcon({ domain }: { domain: string }) {
  const icons: Record<string, string> = {
    finance: '💰', shopping: '🛒', health: '🏥', travel: '✈️',
    productivity: '📋', tech: '💻', home: '🏠', social: '💬',
  };
  return (
    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-xl">
      {icons[domain] ?? '🤖'}
    </div>
  );
}

function ClassBadge({ actionClass, small }: { actionClass: string; small?: boolean }) {
  const colors: Record<string, string> = {
    read_only: 'text-emerald-400 border-emerald-500/20 bg-emerald-500/5',
    reversible: 'text-cyan-400 border-cyan-500/20 bg-cyan-500/5',
    money_out: 'text-amber-400 border-amber-500/20 bg-amber-500/5',
    irreversible: 'text-red-400 border-red-500/20 bg-red-500/5',
    impossible: 'text-red-500 border-red-500/30 bg-red-500/10',
  };
  const cls = colors[actionClass] ?? 'text-slate-400 border-white/10 bg-white/5';
  const size = small ? 'text-[0.5rem] px-1.5 py-0.5' : 'text-[0.6rem] px-2 py-0.5';
  return (
    <span className={`${size} rounded-md border font-mono ${cls}`}>
      {actionClass}
    </span>
  );
}

function MiniStat({ label, value, color = 'slate' }: {
  label: string; value: number | string; color?: string;
}) {
  const colorCls = color === 'emerald' ? 'text-emerald-400' : 'text-slate-200';
  return (
    <div className="px-2.5 py-1.5 rounded-lg bg-white/[0.03] border border-white/5">
      <div className="text-[0.45rem] uppercase tracking-widest text-slate-600">{label}</div>
      <div className={`text-[0.65rem] font-mono font-bold ${colorCls}`}>{value}</div>
    </div>
  );
}
