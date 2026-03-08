"""
Ebbiforge CLI — Command-line interface.

Usage:
    ebbiforge version          Show version and system info
    ebbiforge demo             Run the interactive swarm demo
    ebbiforge benchmark        Run performance benchmark
    ebbiforge example <name>   Run a built-in example
"""

import sys
import os
import runpy


def _version():
    """Show version and system info."""
    from ebbiforge import __version__
    print("Ebbiforge Framework")
    print("=" * 40)
    print(f"  Version:  {__version__}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")

    # Check if Rust core is available
    try:
        import ebbiforge_core
        print(f"  Rust Core: ✅ Available")
    except ImportError:
        print(f"  Rust Core: ❌ Not installed (run: maturin develop)")

    # Check optional deps
    try:
        import requests
        print(f"  Connectors: ✅ requests available")
    except ImportError:
        print(f"  Connectors: ⚠️  Install with: pip install ebbiforge[connectors]")

    print()


def _demo():
    """Run the interactive swarm demo."""
    try:
        import ebbiforge_core as cogops
    except ImportError:
        print("❌ Rust core not available. Build with: maturin develop --release")
        sys.exit(1)

    print("🐝 Ebbiforge Interactive Demo")
    print("=" * 50)
    print("Initializing 10,000 agents...\n")

    swarm = cogops.TensorSwarm(agent_count=10000)
    swarm.register_locations(
        villages=[(100, 200), (400, 500), (700, 300)],
        towns=[(250, 400), (600, 600)],
        cities=[(500, 500)],
        ambush_zones=[(300, 350)],
    )

    import time
    for tick in range(500):
        start = time.time()
        swarm.tick()
        elapsed = (time.time() - start) * 1000

        if tick % 50 == 0:
            health = swarm.health
            mean_h = sum(health) / len(health) if health else 0
            surprise = swarm.surprise_scores
            mean_s = sum(surprise) / len(surprise) if surprise else 0
            print(
                f"  Tick {tick:>4} | "
                f"{elapsed:>5.1f}ms | "
                f"Health: {mean_h:.3f} | "
                f"Surprise: {mean_s:.4f}"
            )

        # Inject a surprise at tick 200
        if tick == 200:
            print("\n  ⚡ INJECTING SURPRISE CASCADE at (500, 500)!\n")
            swarm.apply_environmental_shock(
                location=(500, 500), radius=10, intensity=1.0
            )

    print("\n✅ Demo complete. 500 ticks, 10,000 agents.")
    print("   Run 'ebbiforge example evolution' for evolution demo.")


def _benchmark():
    """Run performance benchmark."""
    try:
        import ebbiforge_core as cogops
    except ImportError:
        print("❌ Rust core not available. Build with: maturin develop --release")
        sys.exit(1)

    import time

    sizes = [1_000, 10_000, 100_000]
    print("🏎️  Ebbiforge Performance Benchmark")
    print("=" * 50)

    for n in sizes:
        swarm = cogops.TensorSwarm(agent_count=n)

        # Warm up
        for _ in range(10):
            swarm.tick()

        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            swarm.tick()
            times.append((time.time() - start) * 1000)

        avg = sum(times) / len(times)
        p99 = sorted(times)[98]
        throughput = n / (avg / 1000)

        print(f"\n  {n:>10,} agents:")
        print(f"    Avg tick:    {avg:>8.2f} ms")
        print(f"    P99 tick:    {p99:>8.2f} ms")
        print(f"    Throughput:  {throughput:>12,.0f} agents/sec")

    print("\n✅ Benchmark complete.")


def _example(name: str):
    """Run a built-in example."""
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

    # Map short names to files
    example_map = {
        "hello": "01_hello_swarm.py",
        "evolution": "02_evolution.py",
        "live": "03_live_data.py",
        "reasoning": "04_selective_reasoning.py",
        "compliance": "05_compliance.py",
    }

    if name not in example_map:
        print(f"Unknown example: '{name}'")
        print(f"Available: {', '.join(example_map.keys())}")
        sys.exit(1)

    filepath = os.path.join(examples_dir, example_map[name])
    if not os.path.exists(filepath):
        print(f"Example file not found: {filepath}")
        sys.exit(1)

    print(f"Running example: {name}\n")
    try:
        runpy.run_path(filepath, run_name="__main__")
    except SystemExit:
        pass  # Allow examples to call sys.exit()
    except Exception as e:
        print(f"\n❌ Example '{name}' failed: {e}")
        sys.exit(1)


def main():
    """CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "version":
        _version()
    elif cmd == "demo":
        _demo()
    elif cmd == "benchmark":
        _benchmark()
    elif cmd == "example":
        if len(args) < 2:
            print("Usage: ebbiforge example <name>")
            print("Available: hello, evolution, live, reasoning, compliance")
            return
        _example(args[1])
    else:
        print(f"Unknown command: '{cmd}'")
        print("Run 'ebbiforge help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
