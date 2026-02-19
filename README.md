# Bedrock Prompt Caching Tutorial

Cut LLM inference costs by up to **90%** using Amazon Bedrock prompt caching and model tier selection.

## What You'll Build

A production-optimized customer support agent that uses:
- **Prompt caching** to eliminate redundant token processing (39–54% cost reduction)
- **Cross-model benchmarking** across Nova Pro, Lite, and Micro
- **CloudWatch observability** to monitor cache hit rates in production

## Project Structure

```
.
├── 01_baseline_no_cache.py          # Baseline: no optimization
├── 02_with_prompt_caching.py        # Add prompt caching (Converse API)
├── run_all_benchmarks.py            # Run all models × baseline/cached
├── product_docs.txt                 # Sample product documentation (~2,069 tokens)
├── results_01_baseline.json         # Nova Pro baseline results
├── results_02_cached.json           # Nova Pro cached results
├── results_full_nova.json           # All 3 Nova models comparison
└── requirements.txt                 # Python dependencies
```

## Prerequisites

- AWS account with Bedrock access enabled in us-east-1
- Model access for Amazon Nova Pro, Nova Lite, and Nova Micro (enabled by default)
- Python 3.11+ with boto3 >= 1.35.76
- AWS CLI v2 configured with `bedrock-runtime:*` permissions

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: Establish baseline costs
python 01_baseline_no_cache.py

# Step 2: See caching in action
python 02_with_prompt_caching.py

# Step 3: Run full comparison across all Nova models
python run_all_benchmarks.py
```

## Benchmark Results (Amazon Nova, 5-turn conversation)

Projected monthly cost at 1,000 conversations/day:

| Model | Baseline | With Caching | Savings |
|-------|----------|-------------|---------|
| Nova Pro | $334.61 | $169.99 | 49% |
| Nova Lite | $30.33 | $18.41 | 39% |
| Nova Micro | $16.99 | $9.47 | 44% |
| **Pro → Micro + cache** | **$334.61** | **$9.47** | **97%** |

## Cost to Run This Tutorial

Under $0.15 for all benchmarks.

