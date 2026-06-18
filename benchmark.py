import argparse, time, json
from pathlib import Path

SUITES = ["llm_inference", "gemm", "flash_attn", "multi_gpu"]

def run_benchmark(suite, gpu_type, iterations=100):
    results = []
    for i in range(iterations):
        start = time.perf_counter()
        # Benchmark kernel execution
        elapsed = time.perf_counter() - start
        results.append(elapsed)
    return {"mean_ms": sum(results)/len(results)*1000, "min_ms": min(results)*1000}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["all"]+SUITES, default="all")
    p.add_argument("--gpu", default="mi300x")
    p.add_argument("--output", default="results/")
    args = p.parse_args()
    suites = SUITES if args.suite == "all" else [args.suite]
    for s in suites:
        print(f"Running {s} on {args.gpu}...")
        r = run_benchmark(s, args.gpu)
        print(f"  mean={r['mean_ms']:.2f}ms  min={r['min_ms']:.2f}ms")
