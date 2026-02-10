# Nano-vLLM-Go

A lightweight LLM inference engine built from scratch in pure Go, implementing vLLM-style scheduling and memory management.

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
# Build the binary
go build -o ask-gpt2 ask_gpt2.go

# Ask a question
./ask-gpt2 "The capital city of France is"
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

Currently supports GPT-2 architecture:
- GPT-2 Small (124M parameters)
- GPT-2 Medium (355M parameters)
- GPT-2 Large (774M parameters)
- GPT-2 XL (1.5B parameters)

## Performance

On Apple M-series (Metal acceleration disabled):
- Prefill: ~8 tokens/second
- Decode: ~6 tokens/second

## Examples

```bash
# Ask about capitals
./ask-gpt2 "The capital city of Italy is"
# Output: Rome

# Complete a sentence
./ask-gpt2 "Once upon a time"
# Output: (story continuation)

# Run quiz demo
./demo_capitals.sh
```

## Project Structure

```
nano-vllm-go/
├── nanovllm/          # Inference engine
├── purego/            # Pure Go tensor ops
│   └── tensor/        # Matrix operations
├── models/            # Downloaded models
├── ask_gpt2.go        # Main binary
└── demo_capitals.sh   # Demo script
```

## Development

```bash
# Run tests
go test ./...

# Build
go build -o ask-gpt2 ask_gpt2.go

# Format
go fmt ./...
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the architecture of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) by GeeeekExplorer.
