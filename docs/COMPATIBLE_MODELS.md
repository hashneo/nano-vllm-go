# Compatible Models

Models that work with nano-vllm-go implementations.

## ✅ Fully Working (Pure Go Implementation)

### GPT-2 Models

All GPT-2 variants are fully supported with pure Go implementation.

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
make ask
./bin/ask gpt2 "The capital city of France is"
# Output: Paris

# Run demo
./demo_capitals.sh
```

**Performance** (Apple M-series):
- Prefill: ~8 tokens/second
- Decode: ~6 tokens/second

---

### Llama 3.2 Models

Llama 3.2 models are fully supported with GQA, RoPE, and SwiGLU implementations.

| Model | Parameters | Size | Context | Download Command |
|-------|------------|------|---------|------------------|
| **Llama 3.2 1B Instruct** | 1B | ~2.5GB | 128K | `python3 scripts/download_model.py --model meta-llama/Llama-3.2-1B-Instruct --output ./models/llama-3.2-1b-instruct` |
| **Llama 3.2 3B Instruct** | 3B | ~6GB | 128K | `python3 scripts/download_model.py --model meta-llama/Llama-3.2-3B-Instruct --output ./models/llama-3.2-3b-instruct` |

**Features**:
- Grouped-Query Attention (GQA) - 32 query heads, 8 KV heads
- RoPE (Rotary Position Embeddings)
- SwiGLU activation function
- RMSNorm layer normalization
- Chat-optimized instruction following

**Important**: Llama models require Python tokenizer for accurate BPE encoding (280K+ merge rules). The `ask` CLI automatically uses the Python tokenizer helper for Llama models.

**Usage**:
```bash
# Download model (requires HuggingFace authentication)
python3 scripts/download_model.py --model meta-llama/Llama-3.2-1B-Instruct --output ./models/llama-3.2-1b-instruct

# Build and run
make ask
./bin/ask llama "What is the capital of France?"
# Output: The capital of France is Paris.

# Ask math questions
./bin/ask llama "What is 2 + 2?"
# Output: 2 + 2 = 4
```

**Performance** (Apple M-series):
- Prefill: ~1.5 tokens/second
- Decode: ~1.7 tokens/second

**Note**: Llama models require correct spelling in prompts for best results. The model is sensitive to typos (e.g., "captial" vs "capital").

---

### Granite 3.0 Models (IBM)

Mixture of Experts architecture with 32 experts and top-8 routing.

| Model | Parameters | Active | Size | Context | Download Command |
|-------|------------|--------|------|---------|------------------|
| **Granite 3.0 1B (350M active)** | 1B | 400M | ~1.5GB | 4K | `python3 scripts/download_model.py --model ibm-granite/granite-3.0-1b-a400m-instruct --output ./models/granite-350m` |

**Features**:
- Mixture of Experts (32 experts, top-8 routing per token)
- Grouped-Query Attention (16 query heads, 8 KV heads)
- RoPE position embeddings
- SwiGLU activation with GLU-style gating
- muP scaling (attention/residual/logits multipliers)
- Efficient: Only ~400M parameters active per forward pass

**Usage**:
```bash
# Download model
python3 scripts/download_model.py --model ibm-granite/granite-3.0-1b-a400m-instruct --output ./models/granite-350m

# Build and run
make ask
./bin/ask granite "What is the capital of Germany?"
# Output: The capital of Germany is Berlin. It has been the capital since...

./bin/ask granite "What is 2 + 2?"
# Output: 2 + 2 equals 4.
```

**Performance** (Apple M-series):
- Prefill: ~2.8 tokens/second
- Decode: ~2.8 tokens/second

**Status**: ✅ Fully working - correctly generates coherent responses

---

## 🧪 Experimental (Generic Implementation)

### Granite 4.0 Models (IBM)

Hybrid architecture with Attention + Mamba2 state space models.

| Model | Parameters | Size | Context | Download Command |
|-------|------------|------|---------|------------------|
| **Granite 4 Nano 350M** | 350M | ~1.3GB | 32K | `python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-4h-350m` |
| **Granite 4 Nano 1B** | 1B | ~3.8GB | 128K | `python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-1b --output ./models/granite-4h-1b --fp16` |

**Features**:
- Hybrid: 4 attention layers + 28 Mamba2 layers
- Grouped-Query Attention (GQA)
- Very long context windows
- Efficient for long sequences

**Status**: Experimental - architecture implemented but not fully tested

---

## 🧪 Partially Tested (Architecture Implemented)

### TinyLlama

| Model | Parameters | Size | Context | Architecture Features |
|-------|------------|------|---------|----------------------|
| **TinyLlama 1.1B Chat** | 1.1B | ~2.2GB | 2048 | GQA, RoPE, SwiGLU, RMSNorm |

Same architecture as Llama 3.2, works with `./bin/ask llama` command.

**Download**:
```bash
python3 scripts/download_model.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --output ./models/tinyllama-1.1b-chat
```

### Mistral Models

| Model | Parameters | Size | Context | Architecture Features |
|-------|------------|------|---------|----------------------|
| **Mistral 7B** | 7B | ~14GB | 8K | GQA, Sliding Window, RoPE |

Architecture implemented but not fully tested.

---

## 📚 Educational Only (Code Available)

These architectures have configuration and loading code for educational purposes, but haven't been fully tested with real models.

### Falcon Models

| Model | Parameters | Size | Context | Architecture Features |
|-------|------------|------|---------|----------------------|
| **Falcon 7B Instruct** | 7B | ~14GB | 2048 | MQA, RoPE, Parallel blocks |

**Features**:
- Multi-Query Attention (71 query heads, 1 KV head)
- RoPE position embeddings
- Parallel attention + FFN blocks
- LayerNorm, GELU activation

**Usage**:
```bash
# Download model
python3 scripts/download_model.py --model tiiuae/falcon-7b-instruct --output ./models/falcon-7b-instruct --fp16

# Build and run
make ask
./bin/ask falcon "What is the capital of Germany?"
```

**Status**: ⚠️ **Experimental - produces garbage output**
- Configuration correctly detects MQA (1 KV head)
- RoPE implementation matches PyTorch for GQA models
- Issue appears specific to Falcon's parallel block architecture
- Debugging in progress

**Known Issues**:
- Generates repetitive tokens or special characters instead of coherent text
- Very slow on CPU (7B parameters, ~0.2 tok/s)
- May require architecture-specific attention or normalization fixes

---

## Model Requirements

### Disk Space
- **GPT-2 Small**: 500MB
- **GPT-2 Medium**: 1.4GB
- **GPT-2 Large**: 3GB
- **GPT-2 XL**: 6GB
- **Llama 3.2 1B**: 2.5GB
- **Llama 3.2 3B**: 6GB
- **TinyLlama 1.1B**: 2.2GB
- **Granite 350M**: 1.3GB
- **Granite 1B**: 3.8GB (with FP16)

### Memory (RAM)
Approximately 2-3× model size during inference:
- **GPT-2 Small**: 1-2GB RAM
- **GPT-2 Medium**: 3-4GB RAM
- **Llama 3.2 1B**: 5-6GB RAM
- **Llama 3.2 3B**: 12-15GB RAM
- **Granite 350M**: 2-3GB RAM

### Format
All models must be in **safetensors** format. The download script automatically converts from HuggingFace format.

### Python Dependencies
For Llama models, you need Python 3 with `transformers` library for accurate tokenization:
```bash
pip install transformers
```

---

## Quick Download Script

Use the interactive downloader:

```bash
chmod +x scripts/download_compatible_models.sh
./scripts/download_compatible_models.sh
```

Or download directly:

```bash
# Recommended for beginners: GPT-2 Small (fast to download, works great)
python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small

# Recommended for chat: Llama 3.2 1B Instruct (requires HuggingFace auth)
python3 scripts/download_model.py --model meta-llama/Llama-3.2-1B-Instruct --output ./models/llama-3.2-1b-instruct

# For experimentation: Granite 350M
python3 scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m
```

---

## Unsupported Models

These models **won't work** without additional implementation:

- **GPT-3/GPT-4**: Not open source
- **Claude**: Not open source
- **Llama 2**: Different architecture specifics
- **Qwen**: Different architecture specifics
- **Gemma**: Different architecture specifics

**Note**: Llama 3.2 and compatible models (TinyLlama) are now fully supported!

To add support for new architectures, implement:
1. Config in `purego/tensor/config.go`
2. Weight mapping in `purego/tensor/generic_loader.go`
3. Any architecture-specific operations (RoPE, MQA, etc.)
4. Tokenizer support in `purego/universal_tokenizer.go` or use Python tokenizer helper

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
