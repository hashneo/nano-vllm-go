# Compatible Models

Models that work with nano-vllm-go implementations.

## ✅ Fully Working (Pure Go Implementation)

### GPT-2 Models

All GPT-2 variants are fully supported with the pure Go tensor implementation.

| Model | Parameters | Size | Context | Download Command |
|-------|------------|------|---------|------------------|
| **GPT-2 Small** | 124M | ~500MB | 1024 | `python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small` |
| **GPT-2 Medium** | 355M | ~1.4GB | 1024 | `python3 scripts/download_model.py --model gpt2-medium --output ./models/gpt2-medium` |
| **GPT-2 Large** | 774M | ~3GB | 1024 | `python3 scripts/download_model.py --model gpt2-large --output ./models/gpt2-large` |
| **GPT-2 XL** | 1.5B | ~6GB | 1024 | `python3 scripts/download_model.py --model gpt2-xl --output ./models/gpt2-xl` |

**Features**:
- Pure Go implementation (no C dependencies)
- KV caching enabled
- BPE tokenization
- Works on any platform

**Usage**:
```bash
# Download model
python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small

# Build and run
make ask-gpt2
./bin/ask-gpt2 "The capital city of France is"
# Output: Paris

# Run demo
./demo_capitals.sh
```

**Performance** (Apple M-series):
- Prefill: ~8 tokens/second
- Decode: ~6 tokens/second

---

## 🧪 Experimental (Generic Implementation)

### Granite Models (IBM)

Hybrid architecture with Attention + Mamba2 state space models.

| Model | Parameters | Size | Context | Download Command |
|-------|------------|------|---------|------------------|
| **Granite 4 Nano 350M** | 350M | ~1.3GB | 32K | `python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m` |
| **Granite 4 Nano 1B** | 1B | ~3.8GB | 128K | `python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-1b --output ./models/granite-1b --fp16` |

**Features**:
- Hybrid: 4 attention layers + 28 Mamba2 layers
- Grouped-Query Attention (GQA)
- Very long context windows
- Efficient for long sequences

**Usage**:
```bash
# Download model
python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m

# Build and run
make generic-runner
./bin/generic-runner
```

**Status**: Experimental - architecture implemented but not fully tested

---

## 📚 Educational Only (Code Available)

These architectures have configuration and loading code for educational purposes, but haven't been tested with real models.

### Falcon Models

| Model | Parameters | Size | Context | Architecture Features |
|-------|------------|------|---------|----------------------|
| **Falcon 7B** | 7B | ~14GB | 2048 | MQA, RoPE, Parallel blocks |
| **Falcon 40B** | 40B | ~80GB | 2048 | MQA, RoPE, Parallel blocks |

**Key Learning**: Multi-Query Attention (MQA) implementation in `purego/tensor/mqa.go`

### Llama Models

| Model | Parameters | Size | Context | Architecture Features |
|-------|------------|------|---------|----------------------|
| **Llama 7B** | 7B | ~13GB | 4096 | GQA, RoPE, SwiGLU, RMSNorm |
| **Llama 13B** | 13B | ~24GB | 4096 | GQA, RoPE, SwiGLU, RMSNorm |

**Key Learning**:
- Grouped-Query Attention configuration
- RMSNorm normalization
- SwiGLU activation

---

## Model Requirements

### Disk Space
- **GPT-2 Small**: 500MB
- **GPT-2 Medium**: 1.4GB
- **GPT-2 Large**: 3GB
- **GPT-2 XL**: 6GB
- **Granite 350M**: 1.3GB
- **Granite 1B**: 3.8GB (with FP16)

### Memory (RAM)
Approximately 2-3× model size during inference:
- **GPT-2 Small**: 1-2GB RAM
- **GPT-2 Medium**: 3-4GB RAM
- **Granite 350M**: 2-3GB RAM

### Format
All models must be in **safetensors** format. The download script automatically converts from HuggingFace format.

---

## Quick Download Script

Use the interactive downloader:

```bash
chmod +x scripts/download_compatible_models.sh
./scripts/download_compatible_models.sh
```

Or download directly:

```bash
# Recommended: GPT-2 Small (fast to download, works great)
python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small

# For experimentation: Granite 350M
python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m
```

---

## Unsupported Models

These models **won't work** without additional implementation:

- **GPT-3/GPT-4**: Not open source
- **Claude**: Not open source
- **LLaMA 2/3**: Requires full GQA + RoPE implementation
- **Mistral**: Requires sliding window attention
- **Qwen**: Different architecture specifics

To add support for new architectures, implement:
1. Config in `purego/tensor/config.go`
2. Weight mapping in `purego/tensor/generic_loader.go`
3. Any architecture-specific operations (RoPE, MQA, etc.)

---

## First Time Setup

```bash
# 1. Clone repository
git clone https://github.com/hashneo/nano-vllm-go
cd nano-vllm-go

# 2. Download GPT-2 Small (recommended for testing)
python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small

# 3. Build binaries
make all

# 4. Run demo
./demo_capitals.sh
```

That's it! The demo will automatically use the downloaded model.
