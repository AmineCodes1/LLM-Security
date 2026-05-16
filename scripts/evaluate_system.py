from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis_rag.config import AppSettings
from aegis_rag.evaluation import RagEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG attack success, false positives, and latency")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--poisoned-ratio", type=float, default=0.5, help="Ratio of poisoned runs")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM calls and simulate answers")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output results as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = AppSettings.from_env()
    evaluator = RagEvaluator(settings)

    results = evaluator.run(
        iterations=args.iterations,
        poisoned_ratio=args.poisoned_ratio,
        dry_run=args.dry_run,
    )

    if args.as_json:
        print(json.dumps(asdict(results), indent=2))
        return

    print("\nEvaluation Results")
    print("==================")
    print("Shield ON")
    print(f"  Total runs:       {results.shield_on.total_runs}")
    print(f"  Poisoned runs:    {results.shield_on.poisoned_runs}")
    print(f"  Attack success:   {results.shield_on.attack_success_rate:.2%}")
    print(f"  False positives:  {results.shield_on.false_positive_rate:.2%}")
    print(f"  Avg latency (ms): {results.shield_on.avg_latency_ms:.1f}")

    print("\nShield OFF")
    print(f"  Total runs:       {results.shield_off.total_runs}")
    print(f"  Poisoned runs:    {results.shield_off.poisoned_runs}")
    print(f"  Attack success:   {results.shield_off.attack_success_rate:.2%}")
    print(f"  False positives:  {results.shield_off.false_positive_rate:.2%}")
    print(f"  Avg latency (ms): {results.shield_off.avg_latency_ms:.1f}")

    print("\nLatency impact")
    print(f"  Delta (ms):       {results.latency_delta_ms:.1f}")
    print(f"  Delta (%):        {results.latency_delta_pct:.1f}%")


if __name__ == "__main__":
    main()
