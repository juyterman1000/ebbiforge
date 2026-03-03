import ebbiforge_core as ors
import time
import sys


def get_memory_usage():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0  # MB
    return 0.0


def run_calibration(count):
    print(f"\n--- CALIBRATING: {count:,} AGENTS ---")
    m0 = get_memory_usage()
    t0 = time.perf_counter()

    swarm = ors.ProductionTensorSwarm(agent_count=count)

    m1 = get_memory_usage()
    t1 = time.perf_counter()

    print(f"   Memory Delta: {m1 - m0:.2f} MB")
    print(f"   Init Time:    {t1 - t0:.2f}s")
    return m1 - m0


if __name__ == "__main__":
    print("EBBIFORGE MEMORY CALIBRATION")

    # 1. Measure 1M agents
    m_1m = run_calibration(1_000_000)

    swarm = ors.ProductionTensorSwarm(agent_count=1_000_000)

    # 2. Predictive scaling for 10M
    est_10m = m_1m * 10
    print(f"\nESTIMATED 10M ACTIVE FOOTPRINT: {est_10m:.2f} MB")

    if est_10m > 28000:
        print("Warning: 10M active agents will likely OOM.")
        print("Strategy: Use 1M Active + 9M Dormant (LOD) for the 10M proof.")
    else:
        print("10M scale looks safe for 32GB RAM.")

    # 3. Test extraction overhead
    print("\nTESTING EXTRACTION OVERHEAD (1M Agents)")
    t2 = time.perf_counter()
    health = swarm.get_all_health()
    t3 = time.perf_counter()
    list_mem = sys.getsizeof(health) / 1024 / 1024
    print(f"   List Extraction Time (1M): {t3 - t2:.4f}s")
    print(f"   Python List Memory:        {list_mem:.2f} MB")
    print(f"   Estimated 10M List Memory: {list_mem * 10:.2f} MB")

    # 4. Test Smart Metrics (O(1) transfer)
    print("\nTESTING SMART METRICS (O(1) Overhead)")
    t4 = time.perf_counter()
    metrics = swarm.sample_population_metrics()
    t5 = time.perf_counter()
    print(f"   Metrics Time: {t5 - t4:.6f}s")
    print(f"   Mean Health (from Rust): {metrics.get('mean_health', 0):.4f}")
