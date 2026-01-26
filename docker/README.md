<div align="center">

# 🐳 INFINICHUNK Docker Images

### **Production-Ready Environments for Scalable Reasoning**

*Optimized, reproducible, and ready for distributed training.*

</div>

---

## 📦 Available Images

We provide two polished production images optimized for differnet backends:

| Dockerfile | Backend | CUDA | Best For |
|------------|---------|------|----------|
| `Dockerfile.app.sglang` | **SGLang** | 12.6 | 🚀 **Highest Throughput** (Recommended) |
| `Dockerfile.app.vllm` | **vLLM** | 12.6 | 🛠️ Compatibility / Legacy |

Both images are built on top of the official `verlai/verl` base and include the full INFINICHUNK runtime stack.

---

## 🛠️ Build Instructions

### Option A: SGLang (Recommended)

SGLang offers superior throughput for chunked rollouts due to efficient KV cache management.

```bash
# Run from the root of the repo
docker build \
  -f docker/Dockerfile.app.sglang \
  -t infinichunk-sglang:latest .
```

### Option B: vLLM

If you prefer the vLLM backend or have specific compatibility needs:

```bash
docker build \
  -f docker/Dockerfile.app.vllm \
  -t infinichunk-vllm:latest .
```

---

## 🏃 Running Containers

### 1. Interactive Development

Mount your current directory to develop with live code changes:

```bash
docker run --gpus all --rm -it \
  --shm-size=32g \
  -v $PWD:/workspace/infinichunk \
  -w /workspace/infinichunk \
  infinichunk-sglang:latest bash
```

### 2. Distributed Training (Multi-GPU)

For launching distributed training, ensure shared memory is set correctly:

```bash
docker run --gpus all --rm -d \
  --name infinichunk-train \
  --shm-size=512g \
  --net=host \
  -v /data:/data \
  infinichunk-sglang:latest \
  bash -c "python3 -m verl.trainer.main_policy_iteration ..."
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `OOM: CUDA out of memory` | Reduce `data.train_batch_size` or decrease `actor_rollout_ref.rollout.gpu_memory_utilization` to `0.7` |
| `Shared memory error` | add `--shm-size=32g` (or larger) to your docker run command |
| `NCCL connection timeout` | Use `--net=host` to avoid docker overlay network overhead |

---

<div align="center">

**[⬅️ Back to Main README](../README.md)**

</div>
