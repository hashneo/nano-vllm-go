# Nano-vLLM-Go

A lightweight LLM inference engine built from scratch in pure Go, implementing vLLM-style scheduling and memory management.

## ⚠️ Disclaimer

**This project is for educational purposes only and is not intended for production use.**

This is a learning implementation designed to demonstrate LLM inference concepts including:
- Transformer architecture implementation from scratch
- KV caching and memory management
- Tokenization and model weight loading
- Attention mechanisms (Multi-Head, Grouped-Query, Multi-Query)

For production LLM inference, please use established frameworks like:
- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference with PagedAttention
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient C++ implementation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Industry-standard library

## Documentation

- [Architecture Guide](docs/ARCHITECTURE_GUIDE.md) - Detailed architecture documentation
- [Compatible Models](docs/COMPATIBLE_MODELS.md) - Supported models and architectures
- [Architectures Available](docs/ARCHITECTURES_AVAILABLE.md) - Available attention mechanisms
- [Recent Bug Fixes](docs/FIX_SUMMARY.md) - SwiGLU fix for Llama models
- [Tokenizer Status](docs/TOKENIZER_TODO.md) - Tokenizer limitations and roadmap
- [Agent Guidelines](docs/AGENTS.md) - Development guidelines for sandbox environment

## Features

- **Pure Go Implementation** - No Python dependencies, runs natively on any platform
- **Continuous Batching** - Dynamic batch composition for optimal resource utilization
- **KV Caching** - Key-Value cache for efficient O(N) generation
- **Block-based Memory Management** - Efficient memory allocation with prefix caching
- **Real Model Support** - Runs actual GPT-2 models from HuggingFace

## Quick Start

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/your-username/nano-vllm-go
cd nano-vllm-go

# Download GPT-2 model
python3 download_model.py --model gpt2 --output ./models/gpt2-small
```

### Build and Run

```bash
# Build all binaries
make all

# Or build individually
make ask-gpt2

# Ask a question
./bin/ask-gpt2 "The capital city of France is"
# Output: Paris

# Run the demo
./demo_capitals.sh
```

## Architecture

### Core Components

- **`nanovllm/`** - Main inference engine
  - `llm_engine.go` - Orchestrates prefill/decode phases
  - `scheduler.go` - Manages sequence scheduling
  - `block_manager.go` - KV cache memory management
  - `sequence.go` - Request lifecycle tracking

- **`purego/`** - Pure Go tensor operations
  - `tensor/` - Matrix operations and transformer layers
  - `bpe_tokenizer.go` - Byte-Pair Encoding tokenization
  - `safetensors.go` - Model weight loading

### How It Works

1. **Tokenization** - Text is converted to token IDs using BPE
2. **Prefill Phase** - Process all prompt tokens at once, cache KV states
3. **Decode Phase** - Generate one token at a time, reusing cached KV states
4. **Sampling** - Apply temperature and select next token from logits
5. **Detokenization** - Convert token IDs back to text

## Model Support

### ✅ Fully Supported

**GPT-2 Family** (Multi-Head Attention):
- GPT-2 Small/Medium/Large/XL (124M - 1.5B parameters)
- Pure Go BPE tokenization
- Fast inference on CPU

**Llama 3.2 Family** (Grouped-Query Attention):
- Llama 3.2 1B/3B Instruct
- TinyLlama 1.1B Chat
- Advanced features: GQA, RoPE, SwiGLU, RMSNorm
- Python tokenizer integration for accurate BPE

### 🧪 Experimental

- Granite 350M/1B (Hybrid Mamba2 + Attention)
- Mistral 7B (architecture implemented)
- Falcon 7B (Multi-Query Attention, code available)

See [docs/COMPATIBLE_MODELS.md](docs/COMPATIBLE_MODELS.md) for detailed information, download instructions, and performance benchmarks.

## Performance

On Apple M-series (no GPU acceleration):

**GPT-2 Small:**
- Prefill: ~8 tokens/second
- Decode: ~6 tokens/second

**Llama 3.2 1B:**
- Prefill: ~1.5 tokens/second
- Decode: ~1.7 tokens/second

## Examples

### Using GPT-2

```bash
# Download and setup
python3 scripts/download_model.py --model gpt2 --output ./models/gpt2-small
make ask-gpt2

# Ask about capitals
./bin/ask-gpt2 "The capital city of Italy is"
# Output: Rome

# Complete sentences
./bin/ask-gpt2 "Once upon a time"
# Output: (story continuation)

# Run quiz demo
./demo_capitals.sh
```

### Using Llama 3.2

```bash
# Download and setup (requires HuggingFace authentication)
python3 scripts/download_model.py --model meta-llama/Llama-3.2-1B-Instruct --output ./models/llama-3.2-1b-instruct
make ask-llama

# Ask questions
./bin/ask-llama "What is the capital of France?"
# Output: The capital of France is Paris.

# Math questions
./bin/ask-llama "What is 2 + 2?"
# Output: 2 + 2 = 4

# General knowledge
./bin/ask-llama "Explain photosynthesis in simple terms"
# Output: (detailed explanation)
```

### Other Tools

```bash
./bin/simple-demo          # Simple tokenizer demo (no models needed)
./bin/generic-runner       # Universal architecture runner
```

## Project Structure

```
nano-vllm-go/
├── cmd/               # Command-line binaries
│   ├── ask-gpt2/      # Main GPT-2 Q&A binary
│   ├── generic-runner/ # Universal architecture runner
│   └── simple-demo/   # Simple tokenizer demo
├── nanovllm/          # Inference engine
├── purego/            # Pure Go tensor ops
│   └── tensor/        # Matrix operations
├── scripts/           # Utilities (model downloader)
├── models/            # Downloaded models (gitignored)
└── demo_capitals.sh   # Demo script
```

## Development

```bash
# Build all binaries
make all

# Run tests
go test ./...

# Clean build artifacts
make clean

# Format
go fmt ./...
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the architecture of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by GeeeekExplorer.
