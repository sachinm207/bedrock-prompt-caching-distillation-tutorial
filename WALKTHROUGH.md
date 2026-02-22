# Prompt Caching on Amazon Bedrock -- Developer Walkthrough

This document explains every part of this project from the ground up. If you have never used Amazon Bedrock, boto3, or even thought about LLM pricing, you will still be able to follow along.

---

## 1. What the Project Does

This project measures how much money you can save on AI API calls by turning on a single feature: **prompt caching**.

The scenario is a customer support chatbot. The bot answers questions about a fictional product called SmartWidget Pro. Every time a customer asks a question, the bot needs to read the same product documentation (~2,069 tokens of text) so it can give accurate answers. Without caching, the AI re-reads that documentation from scratch on every single API call. With caching, the AI reads it once, stores it, and pulls it from memory on subsequent calls at a 90% discount.

The project runs a 5-turn support conversation (five questions, five answers) and tracks exactly how many tokens were used, how much it cost, and how long it took. It does this in two modes -- baseline (no caching) and cached -- and across three different AI models (Nova Pro, Nova Lite, Nova Micro). Then it compares the results.

The bottom line from real benchmark runs:

| Model | Monthly cost (no cache) | Monthly cost (with cache) | Savings |
|-------|------------------------|--------------------------|---------|
| Nova Pro | $334.61 | $169.99 | 49% |
| Nova Lite | $30.33 | $18.41 | 39% |
| Nova Micro | $16.99 | $9.47 | 44% |

Switching from Nova Pro without caching to Nova Micro with caching: **97% reduction** ($334.61 down to $9.47 per month).

---

## 2. Why Prompt Caching Matters

Every time you call an LLM through an API, you pay per token. A token is roughly 3/4 of a word -- so "customer support" is about 2 tokens.

In a typical chatbot, the system prompt (the instructions telling the AI how to behave) plus any reference documents get sent on **every single API call**. If your system prompt and product docs total 2,100 tokens, and a customer asks 5 questions in a conversation, you are paying to process those 2,100 tokens five times. Multiply that by 1,000 conversations a day, and that is 10.5 million tokens per day -- just from re-reading the same instructions.

Prompt caching solves this by storing the system prompt after the first call. On subsequent calls, instead of re-processing those 2,100 tokens at full price, the API reads them from cache at a 90% discount. The first call pays a small write premium (25% extra), but every call after that gets the discount.

The math works out heavily in your favor for any application where the same prompt gets reused -- which is almost every production chatbot, RAG system, or support agent.

---

## 3. Tech Stack

| Technology | What it is | Why it is used here |
|-----------|-----------|-------------------|
| Python 3.11+ | Programming language | The scripts are all Python |
| boto3 | AWS SDK for Python | Sends requests to Amazon Bedrock's API |
| Amazon Bedrock | AWS managed AI service | Hosts the Nova models and handles prompt caching |
| Amazon Nova Pro | Large AI model | High-quality responses, most expensive of the three |
| Amazon Nova Lite | Medium AI model | Good quality, much cheaper than Pro |
| Amazon Nova Micro | Small AI model | Text-only, cheapest, fastest |
| Converse API | Bedrock's unified chat API | The specific API endpoint used -- supports caching via `cachePoint` |
| JSON | Data format | Benchmark results are saved as JSON files |

---

## 4. Project Files

```
.
├── 01_baseline_no_cache.py       # Runs 5-turn conversation WITHOUT caching, tracks cost
├── 02_with_prompt_caching.py     # Same conversation WITH caching, compares to baseline
├── run_all_benchmarks.py         # Tests all 3 Nova models x 2 modes, prints comparison table
├── product_docs.txt              # Fictional product documentation for SmartWidget Pro (~2,069 tokens)
├── requirements.txt              # Python dependency: boto3>=1.35.76
├── results_01_baseline.json      # Saved output from baseline run (Nova Pro)
├── results_02_cached.json        # Saved output from cached run (Nova Pro)
├── results_full_nova.json        # Saved output from full benchmark (all 3 models)
├── README.md                     # Quick-start guide and benchmark results summary
└── WALKTHROUGH.md                # This file
```

---

## 5. File-by-File Breakdown

### 5.1 `product_docs.txt` -- The Reference Material

This is a plain text file containing fictional product documentation for "SmartWidget Pro v4.2." It covers features, pricing, integration guides, troubleshooting, and SLA details. It totals about 2,069 tokens.

This file exists because a real support chatbot would need product documentation to answer questions accurately. The AI reads this file so it can reference real details like pricing tiers ($49/mo Standard, $199/mo Pro, custom Enterprise), API rate limits (1,000 req/min for Standard), and migration steps (v3.x to v4.2).

The file is deliberately long enough to make caching worthwhile. Bedrock requires a minimum of ~1,024 tokens for prompt caching to activate. At ~2,069 tokens, this product doc comfortably exceeds that threshold.

### 5.2 `01_baseline_no_cache.py` -- The "Before" Measurement

This script establishes how much a 5-turn conversation costs without any optimization. It is the control group.

**Imports and setup (lines 1-14):**

```python
import boto3
import json
import time
from pathlib import Path

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
```

- `boto3` is the AWS SDK. It lets Python talk to AWS services.
- `boto3.client("bedrock-runtime", ...)` creates a connection to Bedrock's runtime API in the us-east-1 region. The "runtime" part means we are calling models (inference), not managing them.
- `time` is used to measure how long each API call takes (latency).
- `Path` is used to load `product_docs.txt` from the same directory.

**Model selection (line 18):**

```python
MODEL_ID = "amazon.nova-pro-v1:0"
```

This is the model identifier for Amazon Nova Pro. Bedrock uses these IDs to route your request to the right model. The `:0` at the end is the version number. Other options in this project: `amazon.nova-lite-v1:0` and `amazon.nova-micro-v1:0`.

**System prompt (lines 21-30):**

```python
SYSTEM_PROMPT = """You are a senior customer support agent for SmartWidget, a SaaS company.

Rules:
- Always be polite, professional, and concise
- Reference the product documentation when answering
- If the answer is not in the documentation, say so honestly
- Format responses with bullet points for clarity
- Never make up product features or pricing that isn't documented
- Keep responses under 150 words unless the question requires more detail
"""
```

The system prompt tells the AI how to behave. It acts as permanent instructions that apply to every message in the conversation. This is separate from the user's question.

**Loading and combining docs (lines 33-36):**

```python
PRODUCT_DOCS = Path(__file__).parent.joinpath("product_docs.txt").read_text()
FULL_SYSTEM = SYSTEM_PROMPT + "\n\n--- PRODUCT DOCUMENTATION ---\n\n" + PRODUCT_DOCS
```

`Path(__file__).parent` means "the folder this script lives in." This ensures the script can find `product_docs.txt` regardless of where you run it from.

The system prompt and product docs get concatenated into one big string (`FULL_SYSTEM`). This entire block -- behavioral rules plus product docs -- gets sent on every API call. That is approximately 2,100 tokens re-processed from scratch each time. This is the waste that caching eliminates.

**The test questions (lines 39-45):**

```python
QUESTIONS = [
    "What are the main features of SmartWidget Pro?",
    "How do I configure the API integration? Give me a quick start guide.",
    "What's the pricing for enterprise customers?",
    "My API is returning 429 errors. How do I fix this?",
    "How do I migrate from v3.x to v4.2? What are the breaking changes?",
]
```

Five realistic support questions. They simulate a single customer conversation with five back-and-forth turns. The questions are the same across baseline and cached scripts so the comparison is fair.

**The API call function (lines 48-75):**

```python
def ask_question_baseline(question, conversation_history=None):
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
```

Key things happening here:

- `bedrock.converse(...)` is the Converse API call. This is the actual moment where tokens are consumed and you get billed.
- `system=[{"text": FULL_SYSTEM}]` sends the entire system prompt + product docs. In baseline mode, this is a plain text block with no caching.
- `messages=messages` sends the conversation history plus the new question. The history grows with each turn -- turn 1 has just the question, turn 5 has all previous questions and answers plus the new question.
- `inferenceConfig={"maxTokens": 512, "temperature": 0.1}`:
  - `maxTokens`: caps the response length at 512 tokens. Prevents runaway responses.
  - `temperature`: controls randomness. 0.1 is very low -- the model gives consistent, deterministic answers. For a support bot, you want consistent answers, not creative ones.
- `response["usage"]` contains `inputTokens` and `outputTokens` -- the exact token counts Bedrock processed for this call.

**Conversation loop and history (lines 78-119):**

The `run_baseline()` function loops through all 5 questions. After each turn, it appends both the question and the AI's answer to `history`. This means each subsequent turn sends more data:

- Turn 1: system prompt + docs + question 1 = ~2,140 input tokens
- Turn 2: system prompt + docs + Q1 + A1 + question 2 = ~2,446 input tokens
- Turn 5: system prompt + docs + Q1-Q4 + A1-A4 + question 5 = ~2,860 input tokens

These numbers come from the actual results in `results_01_baseline.json`. The input token count grows each turn because the full conversation history is resent.

**Cost calculation (lines 132-159):**

```python
PRICING = {
    "amazon.nova-pro-v1:0": (0.80, 3.20),
    "amazon.nova-lite-v1:0": (0.06, 0.24),
    "amazon.nova-micro-v1:0": (0.035, 0.14),
}
input_price, output_price = PRICING.get(MODEL_ID, (3.0, 15.0))
input_cost = (total_input_tokens / 1_000_000) * input_price
output_cost = (total_output_tokens / 1_000_000) * output_price
```

Bedrock pricing is per 1 million tokens. For Nova Pro: $0.80 per million input tokens, $3.20 per million output tokens. Output tokens cost more because the model has to generate them (compute-intensive), while input tokens just need to be read and understood.

The monthly projection multiplies the 5-turn session cost by 1,000 conversations/day and 30 days. From the actual baseline run: $0.012814 per session x 1,000 x 30 = **$384.43/month**.

### 5.3 `02_with_prompt_caching.py` -- The "After" Measurement

This script is nearly identical to the baseline. The critical difference is two lines.

**The cache point (lines 53-59):**

```python
response = bedrock.converse(
    modelId=MODEL_ID,
    system=[
        {"text": FULL_SYSTEM},
        {"cachePoint": {"type": "default"}},   # <-- THIS IS THE CHANGE
    ],
    messages=messages,
    inferenceConfig={"maxTokens": 512, "temperature": 0.1},
)
```

The `system` parameter now has two items instead of one:
1. `{"text": FULL_SYSTEM}` -- the same system prompt + docs as before
2. `{"cachePoint": {"type": "default"}}` -- a marker telling Bedrock: "everything above this point should be cached"

That is the entire code change. One additional dictionary in the `system` list.

**What happens at runtime:**

- **Turn 1 (cache WRITE):** Bedrock has never seen this system prompt before. It processes the full ~2,130 tokens, generates the response, and stores the system prompt in its cache. You pay a 25% premium on those cached tokens for the write operation. The `usage` response shows `cacheWriteInputTokens: 2130` and `inputTokens: 10` (just the question itself).
- **Turns 2-5 (cache READ):** Bedrock recognizes the system prompt is already cached. Instead of re-processing ~2,130 tokens, it reads them from cache at a 90% discount. The `usage` response shows `cacheReadInputTokens: ~2,144` and much lower `inputTokens` (only the conversation history and new question).

From the actual results in `results_02_cached.json`:

| Turn | inputTokens | cacheReadTokens | cacheWriteTokens | What happened |
|------|-------------|-----------------|-------------------|--------------|
| 1 | 10 | 0 | 2,130 | Cache WRITE -- first time, paid premium |
| 2 | 286 | 2,143 | 0 | Cache READ -- 90% discount on 2,143 tokens |
| 3 | 416 | 2,144 | 0 | Cache READ |
| 4 | 464 | 2,147 | 0 | Cache READ |
| 5 | 562 | 2,147 | 0 | Cache READ |

Notice that `inputTokens` dropped dramatically. In baseline, turn 1 had 2,140 input tokens. With caching, turn 1 has only 10 input tokens (the non-cached portion) plus 2,130 cache-write tokens. Turns 2-5 show even bigger drops because the system prompt portion is read from cache instead of being counted as regular input.

**Cache-aware cost calculation (lines 158-181):**

```python
PRICING = {
    "amazon.nova-pro-v1:0": {
        "input": 0.80, "output": 3.20,
        "cache_read": 0.08, "cache_write": 1.00,
    },
    ...
}
input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
cache_read_cost = (total_cache_read / 1_000_000) * pricing["cache_read"]
cache_write_cost = (total_cache_write / 1_000_000) * pricing["cache_write"]
output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
```

With caching, there are four cost buckets instead of two:
- **Regular input tokens** ($0.80/M for Nova Pro): tokens NOT covered by the cache -- the conversation history and new questions
- **Cache read tokens** ($0.08/M): tokens pulled from cache -- 90% cheaper than regular input
- **Cache write tokens** ($1.00/M): tokens written to cache on the first call -- 25% more expensive than regular input
- **Output tokens** ($3.20/M): unchanged, the model's response costs the same either way

The cached session cost: $0.006088 per session vs. $0.012814 baseline. Projected monthly: **$182.65 vs. $384.43** -- a 52% reduction.

**Baseline comparison (lines 198-209):**

The script also loads `results_01_baseline.json` (if it exists) and prints a side-by-side comparison. This is why the scripts are designed to run in order -- script 01 creates the baseline file, script 02 reads it.

### 5.4 `run_all_benchmarks.py` -- Full Model Comparison

This script automates what you could do manually by editing `MODEL_ID` in scripts 01 and 02 and running each one. It tests all 3 Nova models in both modes (6 total runs) and prints a comparison table.

**Model list (lines 142-146):**

```python
models = [
    ("amazon.nova-pro-v1:0", "Nova Pro"),
    ("amazon.nova-lite-v1:0", "Nova Lite"),
    ("amazon.nova-micro-v1:0", "Nova Micro"),
]
```

**Unified benchmark function (lines 49-138):**

The `run_benchmark(model_id, use_caching)` function combines the logic from both scripts 01 and 02. The key conditional:

```python
if use_caching:
    system_block = [
        {"text": FULL_SYSTEM},
        {"cachePoint": {"type": "default"}},
    ]
else:
    system_block = [{"text": FULL_SYSTEM}]
```

Same concept -- with caching, add the `cachePoint` marker. Without it, send the plain text.

**Throttle protection (line 167):**

```python
time.sleep(2)
```

A 2-second pause between benchmarks to avoid hitting Bedrock's rate limits. Without this, running 6 benchmarks back-to-back might trigger throttling (HTTP 429 errors).

**Monthly projection formula (line 137):**

```python
"monthly_cost": round(cost * 1000 * 30, 2),
```

Each benchmark runs one 5-turn conversation. Multiply by 1,000 (conversations per day) and 30 (days per month) to project real-world costs. This is the same formula used in scripts 01 and 02.

### 5.5 `requirements.txt`

```
boto3>=1.35.76
```

One dependency. Version 1.35.76 or higher is needed because prompt caching support in the Converse API was added around this version. Earlier versions of boto3 do not recognize the `cachePoint` parameter.

---

## 6. Key Concepts Explained

### Tokens

A token is the basic unit LLMs use to process text. English text averages about 1 token per 0.75 words. So 100 words is roughly 133 tokens. You are billed per token, not per word or per character.

In this project, the system prompt + product docs total about 2,069 tokens. That means every API call starts by paying to process those ~2,069 tokens before even looking at the user's question.

### Prompt Caching

Prompt caching is a feature where the API remembers a portion of your input so it does not have to re-process it from scratch on subsequent calls. It is like a browser caching images -- the first load downloads the image, but subsequent page loads pull it from local storage.

Bedrock's prompt caching stores the prefix of your `system` block. As long as your system prompt does not change between calls, the cache gets reused.

### Cache Point

The `cachePoint` is a marker you insert into the `system` array to tell Bedrock: "cache everything above this." It is a simple dictionary: `{"cachePoint": {"type": "default"}}`. Without this marker, no caching happens. With it, Bedrock automatically manages storage and retrieval.

### Cache Write vs. Cache Read

- **Cache write** happens on the first call (or when the cache has expired). The system prompt is processed at full speed, stored in Bedrock's cache, and you pay a 25% premium on those tokens.
- **Cache read** happens on every subsequent call within the TTL window. The system prompt is pulled from cache instead of being re-processed. You pay only 10% of the normal input price.

From our results: on turn 1, the cache write cost $0.002130 for 2,130 tokens at $1.00/M. On turns 2-5, each cache read cost about $0.000172 for ~2,144 tokens at $0.08/M. That is a 12x price difference per turn.

### TTL (Time to Live)

Bedrock's prompt cache has a 5-minute TTL by default. If more than 5 minutes pass between calls, the cache expires and the next call triggers a fresh cache write. For a support bot handling ongoing conversations, 5 minutes between messages is typical and the cache stays warm.

### Temperature

Temperature controls how random or creative the model's output is. Range: 0.0 to 1.0.

- `temperature: 0.0` -- completely deterministic. Same input always gives the same output.
- `temperature: 0.1` -- very slight randomness. Used in this project because a support bot should give consistent answers.
- `temperature: 1.0` -- maximum randomness. Good for creative writing, bad for customer support.

### Cost Calculation

All pricing is per 1 million tokens. The formula:

```
cost = (token_count / 1,000,000) * price_per_million
```

For example, 12,834 input tokens on Nova Pro:

```
(12,834 / 1,000,000) * $0.80 = $0.010267
```

### Monthly Projections

The scripts assume 1,000 conversations per day, each with 5 turns. One session costs $X, so:

```
monthly_cost = session_cost * 1,000 conversations/day * 30 days
```

This is a rough but reasonable estimate for a mid-size support operation. Adjust the multiplier up or down for your actual volume.

---

## 7. Data Flow Diagram

```
                          BASELINE (no caching)
                          =====================

  +-----------+       +------------------+       +-----------------+
  |  Python   | ----> |  Bedrock API     | ----> |  Nova Pro/      |
  |  Script   |       |  (Converse)      |       |  Lite/Micro     |
  +-----------+       +------------------+       +-----------------+
       |                     |                          |
       |  Sends on EVERY     |  Processes ALL           |  Returns
       |  call:              |  tokens from             |  response +
       |  - system prompt    |  scratch each            |  usage stats
       |    (~2,100 tokens)  |  time                    |  (inputTokens,
       |  - conversation     |                          |   outputTokens)
       |    history           |                          |
       |  - new question     |                          |
       v                     v                          v
  Total input tokens grow each turn: 2,140 -> 2,446 -> 2,638 -> 2,750 -> 2,860



                          WITH CACHING
                          ============

  +-----------+       +------------------+       +-----------------+
  |  Python   | ----> |  Bedrock API     | ----> |  Nova Pro/      |
  |  Script   |       |  (Converse)      |       |  Lite/Micro     |
  +-----------+       +------------------+       +-----------------+
       |                     |                          |
       |  Turn 1:            |  CACHE WRITE:            |
       |  - system prompt    |  Stores ~2,130           |
       |    + cachePoint     |  tokens, charges         |
       |  - question         |  25% premium             |
       |                     |                          |
       |  Turns 2-5:         |  CACHE READ:             |  Returns
       |  - cachePoint ref   |  Reads ~2,144 tokens     |  response +
       |  - conv history     |  from cache at           |  usage stats
       |  - new question     |  90% DISCOUNT            |  (inputTokens,
       |                     |                          |   outputTokens,
       |                     |        +--------+        |   cacheRead,
       |                     |        | CACHE  |        |   cacheWrite)
       |                     +------->| (5 min |<-------+
       |                              |  TTL)  |
       |                              +--------+
       v
  Regular input tokens drop: 10 -> 286 -> 416 -> 464 -> 562
  (Only conversation history + new question -- system prompt is cached)
```

---

## 8. How to Run It Yourself

### Prerequisites

1. **An AWS account** with Bedrock access enabled in the `us-east-1` region.
2. **Model access** for Amazon Nova Pro, Nova Lite, and Nova Micro. These are enabled by default in most accounts -- check the Bedrock Model Access page in the AWS Console.
3. **Python 3.11 or later** installed on your machine.
4. **AWS credentials configured.** Either run `aws configure` to set up your access key and secret, or use an IAM role if you are on an EC2 instance. Your credentials need the `bedrock-runtime:Converse` permission at minimum.

### Step-by-step

**1. Clone the repository:**

```bash
git clone https://github.com/sachinm207/bedrock-prompt-caching-distillation-tutorial.git
cd bedrock-prompt-caching-distillation-tutorial
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

This installs `boto3` version 1.35.76 or higher. If you already have boto3, check your version with `pip show boto3` -- anything below 1.35.76 may not support the `cachePoint` parameter.

**3. Run the baseline benchmark (no caching):**

```bash
python 01_baseline_no_cache.py
```

This runs a 5-turn conversation with Nova Pro and saves results to `results_01_baseline.json`. Takes about 10 seconds. Expected output includes per-turn token counts, latency, and a monthly cost projection.

**4. Run the cached benchmark:**

```bash
python 02_with_prompt_caching.py
```

This runs the same 5-turn conversation with caching enabled. It also loads `results_01_baseline.json` and prints a comparison at the end. You should see cache write tokens on turn 1, and cache read tokens on turns 2-5.

**5. Run the full model comparison (optional):**

```bash
python run_all_benchmarks.py
```

This tests all 3 Nova models in both modes (6 total runs). Takes about 2 minutes due to the 2-second pauses between runs. Prints a comparison table at the end and saves results to `results_full_comparison.json`.

### Changing the model

To test a different model in scripts 01 or 02, change the `MODEL_ID` variable:

```python
MODEL_ID = "amazon.nova-lite-v1:0"    # or "amazon.nova-micro-v1:0"
```

### Cost to run

All three scripts together cost under $0.15 total. Each individual 5-turn conversation costs fractions of a cent.

---

## 9. Glossary

| Term | Definition |
|------|-----------|
| **Amazon Bedrock** | AWS service that provides access to AI models from Amazon and third parties through a unified API. You do not manage any servers -- you send requests and get responses. |
| **boto3** | The official AWS SDK (Software Development Kit) for Python. It lets you interact with AWS services from Python code. |
| **Cache hit** | When the API finds the requested data in its cache and reads it from there instead of processing it fresh. Results in a 90% cost discount on cached tokens. |
| **Cache miss** | When the data is not in the cache (first call, or cache expired). Triggers a cache write. |
| **Cache point** | A marker (`{"cachePoint": {"type": "default"}}`) inserted into the system block to tell Bedrock what to cache. Everything above this marker gets cached. |
| **Cache read tokens** | Tokens pulled from cache on subsequent calls. Billed at 10% of the normal input price (90% discount). |
| **Cache write tokens** | Tokens stored in cache on the first call. Billed at 125% of the normal input price (25% premium). |
| **Converse API** | Bedrock's unified API for chat-style interactions. Supports all Bedrock models with the same interface. Prompt caching is configured through this API's `system` parameter. |
| **Inference** | The process of sending input to an AI model and getting a response. Each inference call costs money based on token counts. |
| **Input tokens** | Tokens in your request (system prompt + conversation history + user question). You pay per input token processed. |
| **Latency** | How long an API call takes from request to response, measured in seconds. |
| **Model ID** | The identifier string Bedrock uses to route requests to a specific model, e.g., `amazon.nova-pro-v1:0`. |
| **Nova Micro** | Amazon's smallest and cheapest text model. Text-only (no image support). $0.035/M input, $0.14/M output. |
| **Nova Lite** | Amazon's mid-tier model. Supports text and images. $0.06/M input, $0.24/M output. |
| **Nova Pro** | Amazon's most capable Nova model. Supports text and images. $0.80/M input, $3.20/M output. |
| **Output tokens** | Tokens in the model's response. These cost more than input tokens because the model has to generate them. |
| **System prompt** | Instructions sent with every API call that tell the model how to behave. Not visible to the user but shapes every response. |
| **Temperature** | A number between 0.0 and 1.0 that controls how random the model's output is. Lower = more deterministic, higher = more creative. |
| **Token** | The basic unit of text that LLMs process. Roughly 3/4 of a word in English. "Hello world" is 2 tokens. You are billed per token. |
| **TTL (Time to Live)** | How long cached data persists before expiring. Bedrock's default TTL for prompt caching is 5 minutes. |
| **Turn** | One question-answer pair in a conversation. A 5-turn conversation means 5 questions and 5 answers. |
