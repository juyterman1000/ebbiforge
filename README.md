<div align="center">
  <img src="logo.svg" alt="OpenRustSwarm" width="280">

  <h3>10 million organisms that react to the crypto market before you read the number.</h3>

  <p>A living swarm intelligence — built in Rust, visualized in WebAssembly, narrated by Gemini.</p>

  <a href="https://github.com/juyterman1000/openrustswarm/stargazers">
    <img src="https://img.shields.io/github/stars/juyterman1000/openrustswarm?style=social" alt="GitHub Stars">
  </a>
  &nbsp;
  <a href="https://github.com/juyterman1000/openrustswarm/fork">
    <img src="https://img.shields.io/github/forks/juyterman1000/openrustswarm?style=social" alt="GitHub Forks">
  </a>

  <br/><br/>

  [![CI](https://github.com/juyterman1000/openrustswarm/actions/workflows/ci.yml/badge.svg)](https://github.com/juyterman1000/openrustswarm/actions)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
  [![Next.js](https://img.shields.io/badge/next.js-15-black.svg)](https://nextjs.org/)
</div>

<p align="center">
  <img src="demo/hero_banner.png" alt="OpenRustSwarm — BTC, ETH, SOL organism clusters reacting to live market data" width="800">
</p>

---

## What is this?

OpenRustSwarm is a biological simulation where autonomous organisms process real-world data as sensory input. It doesn't chart prices — it reacts to them. Biologically.

Each organism has:

- **6 heritable genes** — transfer rate, recovery rate, infection radius, broadcast power, sensitivity, mutation rate
- **SIRS epidemiology** — organisms infect each other with "surprise" when anomalies hit
- **Ebbinghaus memory decay** — they remember shocking events, forget routine ones
- **Spatial hash grid** — O(1) neighbor lookups, real physics, not random
- **Darwinian evolution** — natural selection, crossover, mutation every generation
- **6-channel pheromone field** — danger, trail, food, novelty, alarm, reward

When Bitcoin drops 3%, you don't read a number. You watch organisms die. The survivors evolve higher sensitivity. R0 climbs. The swarm fights back.

> *Can you see it in the organisms before you read the number?*
>
> Yes. That's the point.

---

## Quick Start

```bash
git clone https://github.com/juyterman1000/openrustswarm.git
cd openrustswarm/web
npm install
npm run dev
```

Open `http://localhost:3000` — the swarm starts immediately.

For narration (Gemini speaks as the organism), add to `web/.env.local`:

```env
GEMINI_API_KEY=your_key_here
```

Without it, the swarm still runs — it just can't speak.

---

## Architecture

```
LAYER 5: THE FACE
  Next.js dashboard, WebGL/Canvas2D, crypto ticker,
  R0 tension overlay (red pulse when R0 > 1.2)

LAYER 4: MEMORY
  Ebbinghaus decay, metacognition, curiosity module

LAYER 3: THE HANDS
  OpenClaw integration skill, webhook alerts, signal injection API

LAYER 2: THE VOICE
  Gemini 2.5 Flash narration — speaks as the organism
  "847 organisms near the BTC cluster just died."

LAYER 1: NERVOUS SYSTEM
  100+ Rust source files, WASM bridge
  SIRS, evolution, pheromones, spatial hash, safety shield
```

## Real Data Feeds

The swarm ingests live data every 15 seconds:

| Feed | Source | What organisms feel |
|------|--------|-------------------|
| BTC | CoinGecko | Shockwave in left swarm region |
| ETH | CoinGecko | Shockwave in center region |
| SOL | CoinGecko | Shockwave in right region |
| GitHub | Events API | Activity pulse in center |

Each asset has its own organism cluster. Inter-poll price deltas (not 24h averages) drive reactions.

---

## Performance

Tested at 10M agents (1M active + 9M dormant via 4-tier LOD architecture). See `test_10m_scale.py`.

| Metric | Browser (WASM) | Native (Rust) |
|--------|---------------|---------------|
| Agents | 200,000 @ 60fps | 10,000,000 (LOD) |
| Memory | ~150 MB | 3.71 GB |
| Throughput | real-time rendering | 20.5M updates/sec |
| Per-tick cost | $0.00 | $0.00 |

No LLM in the simulation loop. The biology is pure Rust math.

---

## What's in the repo

```
openrustswarm/
├── openrustswarm-core/    # Rust engine — 100+ source files
│   └── src/
│       ├── swarm/         # SIRS, spatial hash, tensor engine, LOD, mmap
│       ├── evolution/     # Population genetics, sandbox, synthesizer
│       ├── worldmodel/    # Pheromone diffusion, memory consolidation
│       ├── compliance/    # Safety shield, audit trails
│       └── core/          # Workflow, graph orchestration
├── cogops-wasm/           # WASM bridge — compiled and running
├── web/                   # Next.js dashboard (the thing you see)
│   ├── app/api/           # 9 API routes (feeds, narration, OpenClaw)
│   ├── hooks/             # useWasmEngine, useNarration, useRealDataFeed
│   ├── components/swarm/  # AgentCanvas, R0Indicator, NarrationPanel
│   └── public/wasm/       # Compiled WASM binary
├── cogops-skill/          # OpenClaw integration
├── server/                # Python server + swarm brain
├── demo/                  # Demo server
└── examples/              # Usage examples
```

---

## Roadmap

- [x] Rust engine — SIRS, spatial hash, tensor engine, LOD, mmap
- [x] Darwinian evolution — 6 heritable genes, crossover, mutation, natural selection
- [x] WASM bridge — compiled, running in browser at 200K agents
- [x] Real-time data feeds — CoinGecko, GitHub Events
- [x] Gemini narration — organism-voice, threshold-triggered
- [x] R0 tension overlay — visual urgency when R0 > 1.0
- [x] OpenClaw integration — skill, webhook, signal injection
- [ ] Live hosted demo (Vercel)
- [ ] WebAudio — R0 tension should sound tense
- [ ] More data feeds — earthquake, social sentiment, DeFi TVL
- [ ] Mobile-responsive dashboard

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and good first issues.

The easiest entry point: add a new data feed. Each feed is ~50 lines in `web/app/api/feeds/`.

---

## API Routes

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/swarm` | GET | Live swarm metrics |
| `/api/swarm/narrate` | POST | Gemini narration |
| `/api/feeds/crypto` | GET | BTC/ETH/SOL prices |
| `/api/feeds/github` | GET | GitHub activity |
| `/api/openclaw/push` | POST | Forward alerts to OpenClaw |
| `/api/openclaw/inject` | POST/GET | Inject signals into swarm |

---

## License

[MIT License](LICENSE) — use it, fork it, build on it.

---

<div align="center">

  Every star helps this reach developers who want to build something alive.

  <a href="https://github.com/juyterman1000/openrustswarm/stargazers">
    <img src="https://img.shields.io/github/stars/juyterman1000/openrustswarm?style=social" alt="GitHub Stars">
  </a>

  <br/><br/>
  <sub>Built with Rust, WebAssembly, and the belief that organisms are better than dashboards.</sub>
</div>
