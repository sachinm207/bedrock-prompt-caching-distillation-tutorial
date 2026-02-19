"""
Baseline benchmark — no caching, no optimization.
Runs a 5-turn support conversation on Bedrock and records
cost + latency so we have something to compare against.

    python 01_baseline_no_cache.py
"""

import boto3
import json
import time
from pathlib import Path

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# Model to benchmark — change this to test different models
# Nova Pro (works out of the box, no use case form needed)
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

# Realistic multi-turn customer support questions
QUESTIONS = [
    "What are the main features of SmartWidget Pro?",
    "How do I configure the API integration? Give me a quick start guide.",
    "What's the pricing for enterprise customers?",
    "My API is returning 429 errors. How do I fix this?",
    "How do I migrate from v3.x to v4.2? What are the breaking changes?",
]


def ask_question_baseline(question, conversation_history=None):
    """Send a question WITHOUT prompt caching."""
    if conversation_history is None:
        conversation_history = []

    messages = conversation_history + [
        {"role": "user", "content": [{"text": question}]}
    ]

    start_time = time.time()

    response = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": FULL_SYSTEM}],
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
    }


def run_baseline():
    print("=" * 70)
    print(f"BASELINE BENCHMARK — No Caching")
    print(f"Model: {MODEL_ID}")
    print(f"System prompt + docs: ~{len(FULL_SYSTEM.split())} words")
    print("=" * 70)

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    history = []
    turn_results = []

    for i, question in enumerate(QUESTIONS):
        print(f"\n--- Turn {i + 1}/{len(QUESTIONS)} ---")
        print(f"Q: {question}")

        result = ask_question_baseline(question, history)

        print(f"A: {result['text'][:120]}...")
        print(f"  Latency:       {result['latency_s']:.2f}s")
        print(f"  Input tokens:  {result['input_tokens']}")
        print(f"  Output tokens: {result['output_tokens']}")

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]
        total_latency += result["latency_s"]

        turn_results.append({
            "turn": i + 1,
            "question": question,
            "latency_s": result["latency_s"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })

        # Add to conversation history for multi-turn
        history.append({"role": "user", "content": [{"text": question}]})
        history.append({
            "role": "assistant",
            "content": result["response"]["output"]["message"]["content"],
        })

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    print(f"Total turns:         {len(QUESTIONS)}")
    print(f"Total input tokens:  {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total latency:       {total_latency:.2f}s")
    print(f"Avg latency/turn:    {total_latency / len(QUESTIONS):.2f}s")
    print(f"Avg input tokens/turn: {total_input_tokens // len(QUESTIONS)}")

    # Cost estimate — pricing varies by model
    # Nova Pro: $0.80/1M input, $3.20/1M output
    # Verify at https://aws.amazon.com/bedrock/pricing/
    PRICING = {
        "amazon.nova-pro-v1:0": (0.80, 3.20),
        "amazon.nova-lite-v1:0": (0.06, 0.24),
        "amazon.nova-micro-v1:0": (0.035, 0.14),
    }
    input_price, output_price = PRICING.get(MODEL_ID, (3.0, 15.0))
    input_cost = (total_input_tokens / 1_000_000) * input_price
    output_cost = (total_output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    print(f"\n--- Cost Estimate (this session) ---")
    print(f"Input cost:   ${input_cost:.6f}")
    print(f"Output cost:  ${output_cost:.6f}")
    print(f"Total cost:   ${total_cost:.6f}")

    # Projection: 1,000 conversations/day × 5 turns
    daily_input = total_input_tokens * 1000
    daily_output = total_output_tokens * 1000
    daily_cost = total_cost * 1000

    print(f"\n--- Daily Projection (1,000 conversations × 5 turns) ---")
    print(f"Daily input tokens:  {daily_input:,}")
    print(f"Daily output tokens: {daily_output:,}")
    print(f"Daily cost:          ${daily_cost:.2f}")
    print(f"Monthly cost (30d):  ${daily_cost * 30:.2f}")

    # Save results for comparison
    results = {
        "benchmark": "baseline_no_cache",
        "model": MODEL_ID,
        "turns": turn_results,
        "totals": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "latency_s": round(total_latency, 2),
            "cost_usd": round(total_cost, 6),
        },
        "daily_projection": {
            "conversations": 1000,
            "daily_cost_usd": round(daily_cost, 2),
            "monthly_cost_usd": round(daily_cost * 30, 2),
        },
    }

    output_path = Path(__file__).parent / "results_01_baseline.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    run_baseline()
