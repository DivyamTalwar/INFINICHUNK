<div align="center">

# 🚀 INFINICHUNK

### **An attributed VERL/Delethink extension for bounded-context long reasoning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Built on VERL](https://img.shields.io/badge/Built%20on-VERL-purple.svg)](https://github.com/volcengine/verl)
[![Derived from Delethink](https://img.shields.io/badge/Derived%20from-Delethink-orange.svg)](https://github.com/McGill-NLP/the-markovian-thinker)

<br>

<img src="./assets/chunking_engine.svg" width="100%"/>

</div>

---

## 🎯 What is INFINICHUNK?

**INFINICHUNK** is currently an experimental, mechanically renamed derivative of the bounded head-tail carryover implementation published as **Delethink** in [The Markovian Thinker](https://github.com/McGill-NLP/the-markovian-thinker), integrated into a vendored [VERL](https://github.com/volcengine/verl) tree.

> [!IMPORTANT]
> The core loop, trimmers, trainer integration, reward manager, and reproduction scripts are inherited work. The figures below are upstream results and have not yet been independently reproduced by this repository. See [UPSTREAM.md](UPSTREAM.md) and [Notice.txt](Notice.txt). The current package still installs as `verl`; it must not be published as a distinct PyPI package.

### The Quadratic Barrier
Traditional transformer reasoning faces a critical bottleneck: memory and compute scale as **O(N²)**. As reasoning chains grow longer, they become prohibitively expensive.

<p align="center">
  <img src="./assets/complexity_comparison.svg" width="90%"/>
</p>

### The bounded-context approach
The inherited Delethink mechanism breaks generation into **fixed-size chunks** and retains a configured head and tail between chunks. This bounds the active context per iteration; total rollout compute still grows with the number of chunks and serving/runtime details.

<p align="center">
  <img src="./assets/memory_comparison.svg" width="90%"/>
</p>

---

## 🏗️ System Architecture

Our training stack is built for massive scale, leveraging **Ray** for distributed rollout management and **PPO** for policy optimization.

<p align="center">
  <img src="./assets/architecture_animated.svg" width="90%"/>
</p>

### Core Components

| Component | Purpose |
|-----------|---------|
| **Agent Loop** | Manages chunked generation cycles and state carryover logic. |
| **Rollout Engine** | High-throughput async inference using **SGLang** or **vLLM**. |
| **Trainer** | Distributed PPO optimization supporting GRPO and FSDP sharding. |

---

## 📊 Upstream results (not yet reproduced here)

These visualizations summarize results reported by the upstream Delethink project. They are included for reproduction context, not as independent INFINICHUNK measurements.

<p align="center">
  <img src="./assets/results_benchmark.svg" width="100%"/>
</p>

<p align="center">
  <img src="./assets/results_scaling.svg" width="90%"/>
</p>

---

## 🔄 How It Works (Deep Dive)

The core mechanism relies on a "Generate → Trim → Carryover" loop.

<p align="center">
  <img src="./assets/chunking_flow.svg" width="100%"/>
</p>

### Step-by-Step Algorithm
1.  **Initial Generation**: Generate the first chunk with full context `C`.
2.  **Intelligent Trimming**: Keep the `head` (system prompt) and vital `tail` tokens.
3.  **State Carryover**: Append the trimmed state to form the context for the next chunk.
4.  **Recursion**: Repeat until the final answer is reached.

---

## ⚙️ Configuration

### Key Parameters

| Parameter | Symbol | Description | Example |
|-----------|--------|-------------|---------|
| Context Size | `C` | Max tokens per chunk | 8,192 |
| Carryover Size | `m` | Tokens carried between chunks | 4,096 |
| Keep Head | - | Preserved tokens from first response | 100 |

### Recommended Configs

| Config File | Budget | Use Case |
|-------------|--------|----------|
| `r1d-1.5b_deepscaler_infinichunk_24k` | 24K | Standard Math Reasoning |
| `r1d-1.5b_openmath_infinichunk_96k` | 96K | Extended Deep Thought |

---

## 🚀 Quick Start

### 1. Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python=3.10 && source .venv/bin/activate
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[sglang]"
```

### 2. Run Inference Demo

```bash
python infinichunk_tracing_demo.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --infinichunk_context_size 8192
```

### 3. Start Training

```bash
python -m verl.trainer.main_policy_iteration \
  --config-name=r1d-1.5b_deepscaler_infinichunk_24k
```

---

## ❓ FAQ

<details>
<summary><b>Does this work with any model?</b></summary>
Yes! You can plug in almost any base model (Qwen, Llama, Mistral) by changing the <code>model.path</code> in the config.
</details>

<details>
<summary><b>How do I scale to more GPUs?</b></summary>
Export <code>TREETUNEV__NUM_GPUS_PER_NODE=8</code> before running the training script to utilize full nodes.
</details>

---

<div align="center">

**[The Markovian Thinker](https://github.com/McGill-NLP/the-markovian-thinker)** • **[VERL](https://github.com/volcengine/verl)** • **[SGLang](https://github.com/sgl-project/sglang)**

Made with 💜 by Divyam Talwar

</div>
