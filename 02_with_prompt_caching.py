# Same workload as 01_baseline, but with a cachePoint added to the system block.
# Compare the output side-by-side to see cache hits kicking in from Turn 2 onwards.

import boto3
import json
import time
from pathlib import Path

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model to benchmark
MODEL_ID = "amazon.nova-pro-v1:0"


SYSTEM_PROMPT = """You are a senior customer support agent for SmartWidget, a SaaS company.

Rules:
- Always be polite, professional, and concise
- Reference the product documentation when answering
- If the answer is not in the documentation, say so honestly
- Format responses with bullet points for clarity
- Never make up product features or pricing that isn't documented
- Keep responses under 150 words unless the question requires more detail
"""

# Load product docs (~3000 tokens)
PRODUCT_DOCS = Path(__file__).parent.joinpath("product_docs.txt").read_text()

# Combine into full system content
FULL_SYSTEM = SYSTEM_PROMPT + "\n\n--- PRODUCT DOCUMENTATION ---\n\n" + PRODUCT_DOCS

# Same questions as baseline for fair comparison
QUESTIONS = [
    "What are the main features of SmartWidget Pro?",
    "How do I configure the API integration? Give me a quick start guide.",
    "What's the pricing for enterprise customers?",
    "My API is returning 429 errors. How do I fix this?",
    "How do I migrate from v3.x to v4.2? What are the breaking changes?",
]


def ask_question_cached(question, conversation_history=None):
    """Send a question WITH prompt caching enabled."""
    if conversation_history is None:
        conversation_history = []

    messages = conversation_history + [
        {"role": "user", "content": [{"text": question}]}
    ]

    start_time = time.time()

    response = bedrock.converse(
        modelId=MODEL_ID,
        system=[
            {"text": FULL_SYSTEM},
            # Cache point: everything above this is cached for subsequent calls
            {"cachePoint": {"type": "default"}},
        ],
        messages=messages,
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )

    elapsed = time.time() - start_time
    usage = response["usage"]

    return {
        "response": response,
        "text": response["output"]["message"]["content"][0]["text"],
        "latency_s": elapsed,
        "input_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
        "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
    }


def run_cached():
    print("=" * 70)
    print("PROMPT CACHING BENCHMARK")
    print(f"Model: {MODEL_ID}")
    print(f"System prompt + docs: ~{len(FULL_SYSTEM.split())} words")
    print(f"Cache TTL: 5 minutes (default)")
    print("=" * 70)

    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_write = 0
    total_latency = 0.0
    history = []
    turn_results = []

    for i, question in enumerate(QUESTIONS):
        print(f"\n--- Turn {i + 1}/{len(QUESTIONS)} ---")
        print(f"Q: {question}")

        result = ask_question_cached(question, history)

        print(f"A: {result['text'][:120]}...")
        print(f"  Latency:            {result['latency_s']:.2f}s")
        print(f"  Input tokens:       {result['input_tokens']}")
        print(f"  Output tokens:      {result['output_tokens']}")
        print(f"  Cache READ tokens:  {result['cache_read_tokens']}")
        print(f"  Cache WRITE tokens: {result['cache_write_tokens']}")

        if i == 0 and result["cache_write_tokens"] > 0:
            print(f"  ^ First call: cache WRITE (prefix stored)")
        elif result["cache_read_tokens"] > 0:
            print(f"  ^ Cache HIT! Prefix read from cache")

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_cache_read += result["cache_read_tokens"]
        total_cache_write += result["cache_write_tokens"]
        total_latency += result["latency_s"]

        turn_results.append({
            "turn": i + 1,
            "question": question,
            "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cache_read_tokens": result["cache_read_tokens"],
            "cache_write_tokens": result["cache_write_tokens"],
        })

        # Add to conversation history
        history.append({"role": "user", "content": [{"text": question}]})
        history.append({
            "role": "assistant",
            "content": result["response"]["output"]["message"]["content"],
        })

    # Load baseline for comparison
    baseline_path = Path(__file__).parent / "results_01_baseline.json"
    baseline = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)

    # Summary
    print("\n" + "=" * 70)
    print("CACHING SUMMARY")
    print("=" * 70)
    print(f"Total turns:              {len(QUESTIONS)}")
    print(f"Total input tokens:       {total_input_tokens}")
    print(f"Total output tokens:      {total_output_tokens}")
    print(f"Total cache READ tokens:  {total_cache_read}")
    print(f"Total cache WRITE tokens: {total_cache_write}")
    print(f"Total latency:            {total_latency:.2f}s")
    print(f"Avg latency/turn:         {total_latency / len(QUESTIONS):.2f}s")

    # Cost estimate
    # Nova Pro pricing: $0.80/1M input, $3.20/1M output
    # Cache read: 90% discount → $0.08/1M, Cache write: 25% premium → $1.00/1M
    # Verify at https://aws.amazon.com/bedrock/pricing/
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

    pricing = PRICING.get(MODEL_ID, PRICING["amazon.nova-pro-v1:0"])

    # inputTokens from the API already represents non-cached tokens only.
    # cacheReadInputTokens and cacheWriteInputTokens are separate counts.
    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    cache_read_cost = (total_cache_read / 1_000_000) * pricing["cache_read"]
    cache_write_cost = (total_cache_write / 1_000_000) * pricing["cache_write"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost

    print(f"\n--- Cost Breakdown (this session) ---")
    print(f"Non-cached input:  {total_input_tokens:,} tokens × ${pricing['input']}/1M = ${input_cost:.6f}")
    print(f"Cache read:        {total_cache_read:,} tokens × ${pricing['cache_read']}/1M = ${cache_read_cost:.6f}")
    print(f"Cache write:       {total_cache_write:,} tokens × ${pricing['cache_write']}/1M = ${cache_write_cost:.6f}")
    print(f"Output:            {total_output_tokens:,} tokens × ${pricing['output']}/1M = ${output_cost:.6f}")
    print(f"Total cost:        ${total_cost:.6f}")

    # Projection
    daily_cost = total_cost * 1000

    print(f"\n--- Daily Projection (1,000 conversations × 5 turns) ---")
    print(f"Daily cost:          ${daily_cost:.2f}")
    print(f"Monthly cost (30d):  ${daily_cost * 30:.2f}")

    # Comparison with baseline
    if baseline:
        bl = baseline["totals"]
        bl_daily = baseline["daily_projection"]

        lat_reduction = ((bl["latency_s"] - total_latency) / bl["latency_s"]) * 100
        cost_reduction = ((bl["cost_usd"] - total_cost) / bl["cost_usd"]) * 100
        monthly_savings = bl_daily["monthly_cost_usd"] - (daily_cost * 30)

        print(f"\n--- vs BASELINE (no caching) ---")
        print(f"Latency:  {bl['latency_s']:.2f}s → {total_latency:.2f}s ({lat_reduction:+.1f}%)")
        print(f"Cost:     ${bl['cost_usd']:.6f} → ${total_cost:.6f} ({cost_reduction:+.1f}%)")
        print(f"Monthly:  ${bl_daily['monthly_cost_usd']:.2f} → ${daily_cost * 30:.2f} (save ${monthly_savings:.2f}/mo)")

    # Save results
    results = {
        "benchmark": "with_prompt_caching",
        "model": MODEL_ID,
        "turns": turn_results,
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cache_read_tokens": total_cache_read,
            "cache_write_tokens": total_cache_write,
            "latency_s": round(total_latency, 2),
            "cost_usd": round(total_cost, 6),
        },
        "daily_projection": {
            "conversations": 1000,
            "daily_cost_usd": round(daily_cost, 2),
            "monthly_cost_usd": round(daily_cost * 30, 2),
        },
    }

    output_path = Path(__file__).parent / "results_02_cached.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_cached()
