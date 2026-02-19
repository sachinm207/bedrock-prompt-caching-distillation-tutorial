"""Runs every Nova model (Pro, Lite, Micro) with and without caching
and prints a comparison table at the end. Takes ~2 minutes total."""

import boto3
import json
import time
from pathlib import Path

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

SYSTEM_PROMPT = """You are a senior customer support agent for SmartWidget, a SaaS company.

Rules:
- Always be polite, professional, and concise
- Reference the product documentation when answering
- If the answer is not in the documentation, say so honestly
- Format responses with bullet points for clarity
- Never make up product features or pricing that isn't documented
- Keep responses under 150 words unless the question requires more detail
"""

PRODUCT_DOCS = Path(__file__).parent.joinpath("product_docs.txt").read_text()
FULL_SYSTEM = SYSTEM_PROMPT + "\n\n--- PRODUCT DOCUMENTATION ---\n\n" + PRODUCT_DOCS

QUESTIONS = [
    "What are the main features of SmartWidget Pro?",
    "How do I configure the API integration? Give me a quick start guide.",
    "What's the pricing for enterprise customers?",
    "My API is returning 429 errors. How do I fix this?",
    "How do I migrate from v3.x to v4.2? What are the breaking changes?",
]

PRICING = {
    "amazon.nova-pro-v1:0": {
        "input": 0.80, "output": 3.20,
        "cache_read": 0.08, "cache_write": 1.00,
    },
    "amazon.nova-lite-v1:0": {
        "input": 0.06, "output": 0.24,
        "cache_read": 0.006, "cache_write": 0.075,
    },
    "amazon.nova-micro-v1:0": {
        "input": 0.035, "output": 0.14,
        "cache_read": 0.0035, "cache_write": 0.044,
    },
}


def run_benchmark(model_id, use_caching):
    """Run a 5-turn conversation benchmark."""
    label = f"{'CACHED' if use_caching else 'BASELINE'}"
    history = []
    turns = []
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_latency = 0.0

    for i, question in enumerate(QUESTIONS):
        messages = history + [{"role": "user", "content": [{"text": question}]}]

        if use_caching:
            system_block = [
                {"text": FULL_SYSTEM},
                {"cachePoint": {"type": "default"}},
            ]
        else:
            system_block = [{"text": FULL_SYSTEM}]

        start = time.time()
        response = bedrock.converse(
            modelId=model_id,
            system=system_block,
            messages=messages,
            inferenceConfig={"maxTokens": 512, "temperature": 0.1},
        )
        elapsed = time.time() - start

        usage = response["usage"]
        cr = usage.get("cacheReadInputTokens", 0)
        cw = usage.get("cacheWriteInputTokens", 0)

        turns.append({
            "turn": i + 1,
            "latency_s": elapsed,
            "input_tokens": usage["inputTokens"],
            "output_tokens": usage["outputTokens"],
            "cache_read": cr,
            "cache_write": cw,
        })

        total_input += usage["inputTokens"]
        total_output += usage["outputTokens"]
        total_cache_read += cr
        total_cache_write += cw
        total_latency += elapsed

        history.append({"role": "user", "content": [{"text": question}]})
        history.append({
            "role": "assistant",
            "content": response["output"]["message"]["content"],
        })

        cache_status = ""
        if use_caching:
            cache_status = f" | cache_r={cr} cache_w={cw}"
        print(f"  Turn {i+1}: {elapsed:.2f}s | in={usage['inputTokens']} out={usage['outputTokens']}{cache_status}")

    # Calculate cost
    p = PRICING.get(model_id, PRICING["amazon.nova-pro-v1:0"])
    if use_caching:
        # inputTokens from API already represents non-cached tokens only
        cost = (
            (total_input / 1e6) * p["input"]
            + (total_cache_read / 1e6) * p["cache_read"]
            + (total_cache_write / 1e6) * p["cache_write"]
            + (total_output / 1e6) * p["output"]
        )
    else:
        cost = (
            (total_input / 1e6) * p["input"]
            + (total_output / 1e6) * p["output"]
        )

    return {
        "model": model_id,
        "caching": use_caching,
        "turns": turns,
        "total_input": total_input,
        "total_output": total_output,
        "total_cache_read": total_cache_read,
        "total_cache_write": total_cache_write,
        "total_latency": round(total_latency, 2),
        "avg_latency": round(total_latency / len(QUESTIONS), 2),
        "cost": round(cost, 6),
        "monthly_cost": round(cost * 1000 * 30, 2),
    }


def main():
    models = [
        ("amazon.nova-pro-v1:0", "Nova Pro"),
        ("amazon.nova-lite-v1:0", "Nova Lite"),
        ("amazon.nova-micro-v1:0", "Nova Micro"),
    ]

    all_results = {}

    for model_id, name in models:
        for use_caching in [False, True]:
            mode = "cached" if use_caching else "baseline"
            key = f"{name}_{mode}"
            print(f"\n{'='*60}")
            print(f"  {name} — {'WITH CACHING' if use_caching else 'NO CACHING'}")
            print(f"{'='*60}")

            try:
                result = run_benchmark(model_id, use_caching)
                all_results[key] = result
                print(f"  TOTAL: {result['total_latency']}s | ${result['cost']:.6f} | "
                      f"monthly=${result['monthly_cost']}")
            except Exception as e:
                print(f"  SKIPPED — {str(e)[:100]}")

            # Small delay between benchmarks to avoid throttling
            time.sleep(2)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("FULL COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'Latency':>8} {'Input Tok':>10} {'Cache R':>8} {'Cost':>10} {'Monthly':>10}")
    print("-" * 80)

    for key, r in all_results.items():
        print(f"{key:<35} {r['total_latency']:>7.2f}s {r['total_input']:>10,} "
              f"{r['total_cache_read']:>8,} ${r['cost']:>9.6f} ${r['monthly_cost']:>9.2f}")

    # Calculate improvements
    print(f"\n{'='*80}")
    print("IMPROVEMENTS FROM CACHING")
    print(f"{'='*80}")

    for model_id, name in models:
        bl = all_results[f"{name}_baseline"]
        ca = all_results[f"{name}_cached"]

        lat_imp = ((bl["total_latency"] - ca["total_latency"]) / bl["total_latency"]) * 100
        cost_imp = ((bl["cost"] - ca["cost"]) / bl["cost"]) * 100
        monthly_save = bl["monthly_cost"] - ca["monthly_cost"]

        print(f"\n{name}:")
        print(f"  Latency:  {bl['total_latency']}s → {ca['total_latency']}s ({lat_imp:+.1f}%)")
        print(f"  Cost:     ${bl['cost']:.6f} → ${ca['cost']:.6f} ({cost_imp:+.1f}%)")
        print(f"  Monthly:  ${bl['monthly_cost']:.2f} → ${ca['monthly_cost']:.2f} (save ${monthly_save:.2f}/mo)")

    # Save all results
    output_path = Path(__file__).parent / "results_full_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
