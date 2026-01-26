<div align="center">

# 📦 INFINICHUNK Installation Guide

### Fast, Reproducible Setup for Mac, Linux, and Docker

</div>

---

## ⚡ Quick Setup (Recommended for Mac/Linux)

We strongly recommend using `uv` for blazing fast dependency management.

### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-org/INFINICHUNK.git
cd INFINICHUNK

# Create and activate venv (Python 3.10 is required)
uv venv --python=3.10
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install PyTorch with CUDA 12.4 support
uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install INFINICHUNK with SGLang backend
uv pip install -e ".[sglang]"
```

> **Note:** If you are on Mac (MPS), PyTorch will be installed with MPS support automatically.

---

## 🐳 Docker Setup (Production Ready)

For the most stable, reproducible environment, use our optimized Docker image.

### 1. Build the Image
```bash
docker build -f docker/Dockerfile.app.sglang -t infinichunk-sglang .
```

### 2. Run Container
```bash
docker run --gpus all --rm -it \
    --shm-size=32g \
    -v $PWD:/workspace/infinichunk \
    -w /workspace/infinichunk \
    infinichunk-sglang bash
```

---

## 🔧 Advanced Installation

### Without uv (Standard pip)

```bash
conda create -n infinichunk python=3.10 -y
conda activate infinichunk

# Install PyTorch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install package
pip install -e ".[sglang]"
```

### vLLM Backend Support
If you prefer vLLM over SGLang:

```bash
uv pip install -e ".[vllm]"
```

### Flash Attention
For maximum performance on heavy workloads:

```bash
uv pip install flash-attn --no-build-isolation
```

---

## ✅ Verification

Run the tracing demo to ensure everything is working correctly:

```bash
python3 infinichunk_tracing_demo.py --debug=True --samples=1
```

If you see the model answering a math problem chunk-by-chunk, you're ready to go! 🚀
