# Getting Started with Nano-vLLM-Go

This guide will help you get up and running with nano-vllm-go.

## Prerequisites

- Go 1.22 or later
- Basic understanding of LLM inference
- (Optional) Python with nano-vllm for comparison

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nano-vllm-go.git
cd nano-vllm-go
```

### 2. Install dependencies

```bash
go mod download
```

### 3. Build the project

```bash
make build
```

Or manually:

```bash
go build -o bin/example ./example
go build -o bin/bench ./bench
```

## Quick Start

### Using the Mock Implementation

The project includes a mock model runner for testing the architecture without a real model:

```bash
# Run the example
./bin/example
```

This will generate mock outputs using the scheduling and memory management logic.

### Project Structure

```
nano-vllm-go/
├── nanovllm/              # Core library
│   ├── config.go          # Configuration
│   ├── sampling_params.go # Sampling parameters
│   ├── sequence.go        # Sequence management
│   ├── block_manager.go   # KV cache management
│   ├── scheduler.go       # Batch scheduling
│   ├── model_runner.go    # Model interface + mock
│   ├── llm_engine.go      # Main engine
│   ├── llm.go             # User-facing API
│   └── *_test.go          # Tests
├── example/               # Example usage
├── bench/                 # Benchmark tool
├── README.md              # Overview
├── ARCHITECTURE.md        # Detailed architecture
├── INTEGRATION.md         # Integration guide
└── GETTING_STARTED.md     # This file
```

## Basic Usage

### 1. Create a simple program

Create `main.go`:

```go
package main

import (
    "fmt"
    "log"
    "github.com/your-username/nano-vllm-go/nanovllm"
)

func main() {
    // Create config
    config := nanovllm.NewConfig(
        ".",  // Model path (current dir for mock)
        nanovllm.WithMaxNumSeqs(256),
        nanovllm.WithEnforceEager(true),
    )

    // Create LLM (uses mock implementation)
    llm := nanovllm.NewLLM(config)
    defer llm.Close()

    // Set sampling parameters
    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(50),
    )

    // Generate
    prompts := []string{
        "Hello, world!",
        "How are you?",
    }

    outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
    if err != nil {
        log.Fatal(err)
    }

    // Print results
    for i, output := range outputs {
        fmt.Printf("Prompt: %s\n", prompts[i])
        fmt.Printf("Output: %s\n", output.Text)
        fmt.Printf("Tokens: %d\n\n", len(output.TokenIDs))
    }
}
```

### 2. Run it

```bash
go run main.go
```

## Configuration Options

### Config Options

```go
config := nanovllm.NewConfig(
    "/path/to/model",
    nanovllm.WithMaxNumBatchedTokens(16384),  // Max tokens per batch
    nanovllm.WithMaxNumSeqs(512),             // Max concurrent sequences
    nanovllm.WithMaxModelLen(4096),           // Max sequence length
    nanovllm.WithGPUMemoryUtilization(0.9),   // GPU memory usage
    nanovllm.WithTensorParallelSize(1),       // Number of GPUs
    nanovllm.WithEnforceEager(true),          // Disable CUDA graphs
    nanovllm.WithKVCacheBlockSize(256),       // Block size in tokens
    nanovllm.WithNumKVCacheBlocks(1024),      // Number of blocks
    nanovllm.WithEOS(2),                      // EOS token ID
)
```

### Sampling Parameters

```go
samplingParams := nanovllm.NewSamplingParams(
    nanovllm.WithTemperature(0.7),    // Sampling temperature
    nanovllm.WithMaxTokens(100),       // Max tokens to generate
    nanovllm.WithIgnoreEOS(false),     // Whether to ignore EOS
)
```

### Different Sampling Params per Prompt

```go
prompts := []string{"prompt1", "prompt2"}
samplingParams := []*nanovllm.SamplingParams{
    nanovllm.NewSamplingParams(nanovllm.WithTemperature(0.5)),
    nanovllm.NewSamplingParams(nanovllm.WithTemperature(0.9)),
}

outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
```

## Testing

### Run all tests

```bash
make test
```

Or:

```bash
go test ./...
```

### Run with coverage

```bash
make coverage
```

This generates `coverage.html` showing test coverage.

### Run specific tests

```bash
go test -v -run TestBlockManager ./nanovllm
go test -v -run TestSequence ./nanovllm
```

## Benchmarking

Run the benchmark tool:

```bash
./bin/bench
```

This simulates 256 requests with varying input/output lengths and reports throughput.

**Note**: The benchmark uses mock inference, so it only measures scheduling overhead, not actual model performance.

## Next Steps

### 1. Understand the Architecture

Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand:
- How continuous batching works
- How prefix caching saves memory
- How the scheduler coordinates inference
- How block-based memory management works

### 2. Integrate Real Inference

Read [INTEGRATION.md](INTEGRATION.md) for guides on:
- Using HTTP-based inference servers
- CGo with ONNX Runtime
- Pure Go solutions
- Tokenizer integration

### 3. Implement Your Backend

Choose an approach:

**Easiest**: HTTP server
```go
modelRunner := NewHTTPModelRunner("http://localhost:8000")
tokenizer := NewHTTPTokenizer("http://localhost:8000", 2)
llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
```

**Most flexible**: CGo with PyTorch/ONNX
```go
modelRunner := NewCGoModelRunner(config)
tokenizer := NewSentencePieceTokenizer("tokenizer.model")
llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
```

### 4. Experiment and Tune

Try different configurations:

```go
// High throughput, higher latency
config := nanovllm.NewConfig(
    modelPath,
    nanovllm.WithMaxNumSeqs(512),
    nanovllm.WithMaxNumBatchedTokens(32768),
)

// Low latency, lower throughput
config := nanovllm.NewConfig(
    modelPath,
    nanovllm.WithMaxNumSeqs(32),
    nanovllm.WithMaxNumBatchedTokens(4096),
)

// Memory constrained
config := nanovllm.NewConfig(
    modelPath,
    nanovllm.WithGPUMemoryUtilization(0.7),
    nanovllm.WithNumKVCacheBlocks(512),
)
```

## Common Issues

### "model directory does not exist"

The config validates that the model path exists. For the mock implementation, just use `"."`:

```go
config := nanovllm.NewConfig(".")
```

### "kvcache_block_size must be divisible by 256"

Block size must be a multiple of 256:

```go
config := nanovllm.NewConfig(
    ".",
    nanovllm.WithKVCacheBlockSize(256),  // ✓ Valid
    // nanovllm.WithKVCacheBlockSize(200), // ✗ Invalid
)
```

### "greedy sampling is not permitted"

Temperature must be > 0:

```go
samplingParams := nanovllm.NewSamplingParams(
    nanovllm.WithTemperature(0.7),  // ✓ Valid
    // nanovllm.WithTemperature(0.0),  // ✗ Invalid
)
```

## Examples

### Batch Processing

```go
// Process many prompts efficiently
prompts := make([]string, 100)
for i := range prompts {
    prompts[i] = fmt.Sprintf("Prompt %d", i)
}

outputs, _ := llm.GenerateSimple(prompts, samplingParams, true)
```

### Streaming (Not Yet Implemented)

For streaming support, you would extend the engine:

```go
// Future feature
ch := llm.GenerateStream(prompts, samplingParams)
for token := range ch {
    fmt.Print(token.Text)
}
```

### Custom Model Runner

```go
type MyModelRunner struct {
    // Your fields
}

func (m *MyModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    // Your inference logic
    return tokenIDs, nil
}

func (m *MyModelRunner) Close() error {
    // Cleanup
    return nil
}

// Use it
modelRunner := &MyModelRunner{}
llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
```

## Additional Resources

- [nano-vllm Python version](https://github.com/GeeeekExplorer/nano-vllm) - Original implementation
- [vLLM Paper](https://arxiv.org/abs/2309.06180) - Research paper
- [PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) - Blog post explaining the concept

## Getting Help

- Read the architecture documentation
- Check the integration guide
- Look at the test files for usage examples
- Open an issue on GitHub

## Contributing

Contributions welcome! Areas that need work:

- Additional model runner implementations
- Tokenizer integrations
- Performance optimizations
- More comprehensive tests
- Documentation improvements
- Examples with real models

## License

MIT License - see LICENSE file
