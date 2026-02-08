# Nano-vLLM-Go

A Go implementation of nano-vLLM, a lightweight LLM inference engine built from scratch.

## Overview

This is a Go port of [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), focusing on the core scheduling and memory management logic that makes vLLM efficient. The architecture demonstrates:

- **Continuous Batching** - Dynamic batch composition for optimal GPU utilization
- **Prefix Caching** - KV cache block sharing using content-based hashing
- **Block-based Memory Management** - Efficient memory allocation with reference counting
- **Scheduler** - Intelligent prefill/decode phase management
- **Sequence Management** - Request lifecycle handling with state tracking

## Architecture

The codebase is organized into several key components:

- `config.go` - Configuration parameters for the engine
- `sampling_params.go` - Sampling configuration (temperature, max tokens, etc.)
- `sequence.go` - Represents individual generation requests
- `block_manager.go` - Manages KV cache blocks with prefix caching
- `scheduler.go` - Schedules sequences for prefill and decode phases
- `model_runner.go` - Interface for model inference (implement with your backend)
- `llm_engine.go` - Main engine that orchestrates everything
- `llm.go` - User-facing API

## Installation

```bash
go get github.com/your-username/nano-vllm-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/your-username/nano-vllm-go/nanovllm"
)

func main() {
    // Create LLM engine
    config := nanovllm.NewConfig("/path/to/model")
    llm := nanovllm.NewLLM(config)
    defer llm.Close()

    // Set up sampling parameters
    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.6),
        nanovllm.WithMaxTokens(256),
    )

    // Generate
    prompts := []string{
        "Hello, Nano-vLLM-Go!",
        "What is the meaning of life?",
    }

    outputs := llm.Generate(prompts, samplingParams, true)

    for i, output := range outputs {
        fmt.Printf("Prompt %d: %s\n", i, prompts[i])
        fmt.Printf("Output: %s\n\n", output.Text)
    }
}
```

## Key Features

### Continuous Batching
Unlike static batching, sequences can be added and removed from batches dynamically as they complete, maximizing throughput.

### Prefix Caching
Common prompt prefixes share KV cache blocks using xxhash-based content addressing. This dramatically reduces memory usage for similar prompts.

### Memory Management
Block-based KV cache allocation with reference counting enables safe sharing of cached blocks between sequences.

### Scheduler
Separates prefill (processing input tokens) and decode (generating output tokens) phases for optimal batching.

## Implementation Notes

This Go implementation focuses on the scheduling and memory management logic. The actual model inference is abstracted through the `ModelRunner` interface, which you can implement using:

- **CGo bindings** to PyTorch/ONNX Runtime
- **Go ML libraries** like gorgonia or tensorflow-go
- **HTTP/gRPC** calls to Python-based inference servers
- **Custom CUDA kernels** via CGo

The current implementation includes a mock model runner for demonstration purposes.

## Differences from Python Version

- **Model Inference**: Abstracted as an interface instead of direct PyTorch calls
- **Tokenization**: Requires external tokenizer (BPE, SentencePiece, etc.)
- **Tensor Parallelism**: Simplified - production use would need distributed coordination
- **Error Handling**: Idiomatic Go error handling instead of Python exceptions
- **Concurrency**: Go channels and goroutines instead of Python multiprocessing

## License

MIT License - see LICENSE file for details

## Acknowledgments

Based on the excellent [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) project by Xingkai Yu.
