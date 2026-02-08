# ONNX Implementation Guide

This guide shows you how to use nano-vllm-go with real models using ONNX Runtime - all in pure Go!

## Overview

**Architecture:**
```
┌──────────────────────┐
│   Go Application     │
│                      │
│  ┌────────────────┐  │
│  │ nano-vllm-go   │  │  ← Scheduler (continuous batching, memory mgmt)
│  │   (Scheduler)  │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────┴───────┐  │
│  │ ONNX Runtime   │  │  ← Inference engine (runs model)
│  │   (C library)  │  │
│  └────────────────┘  │
└──────────────────────┘
```

## What You Get

✅ **Real Model Inference** - Load and run actual LLM models
✅ **Pure Go** - No Python at runtime
✅ **Cross-platform** - Works on Linux, macOS, Windows
✅ **Fast Setup** - One-time model export, then pure Go
✅ **Production Ready** - Single binary deployment

## Prerequisites

**For Model Export (one-time):**
- Python 3.8+
- `pip install torch transformers onnx`

**For Running:**
- Go 1.21+
- ONNX Runtime library (auto-downloaded by Go package)

## Step 1: Export Model to ONNX (One-time)

Export a HuggingFace model to ONNX format:

```bash
cd ~/Development/github/nano-vllm-go

# Export Qwen2-0.5B (recommended, ~600MB)
python3 scripts/export_to_onnx.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --output ./models/qwen2-onnx

# Or other models:
# TinyLlama (2GB)
python3 scripts/export_to_onnx.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output ./models/tinyllama-onnx

# Phi-2 (5GB)
python3 scripts/export_to_onnx.py \
  --model microsoft/phi-2 \
  --output ./models/phi2-onnx
```

This creates:
```
models/qwen2-onnx/
├── model.onnx          # ONNX model file
├── tokenizer.json      # Tokenizer vocabulary
├── tokenizer_config.json
└── model_info.json     # Metadata
```

## Step 2: Build the Go Program

```bash
# Build ONNX example
go build -o bin/onnx_test ./purego/example_onnx
```

## Step 3: Test with Real Questions

```bash
# Use the exported model
export MODEL_PATH=./models/qwen2-onnx/model.onnx
export TOKENIZER_PATH=./models/qwen2-onnx

# Ask a question
./bin/onnx_test "What is the capital of France?"

# Multiple questions
./bin/onnx_test \
  "What is AI?" \
  "What is ML?" \
  "What is DL?"
```

## Expected Output

```
Nano-vLLM-Go - Real Model Test
================================

Loading model config from: ./models/qwen2-onnx/model_info.json
✓ Model config loaded
  Vocab size: 151936
  EOS token: 151643
  Model: ./models/qwen2-onnx/model.onnx

Loading ONNX model...
✓ ONNX runtime initialized (model will be loaded per-request)
✓ Model loaded
Loading tokenizer...
✓ Loaded HF tokenizer (vocab: 151936, EOS: 151643)
✓ Tokenizer loaded

Generating responses...
Questions: 1

Generating [Prefill: 89tok/s, Decode: 23tok/s] 100% [====] (1/1, 1.2 it/s)

═══════════════════════════════════════════════════════════════════
RESULTS
═══════════════════════════════════════════════════════════════════

📝 Question 1: What is the capital of France?
💬 Answer: The capital of France is Paris.
📊 Tokens: 8

═══════════════════════════════════════════════════════════════════
✓ Test complete!
```

## How It Works

### Model Runner (ONNX)

The ONNX runner loads the model and runs inference:

```go
// purego/onnx_runner.go
runner, err := purego.NewONNXModelRunner(modelPath, config)
runner.Run(sequences, isPrefill)  // Returns next tokens
```

### Tokenizer (HuggingFace)

The HF tokenizer loads from exported files:

```go
// purego/tokenizer.go
tokenizer, err := purego.NewHFTokenizer(tokenizerDir)
tokens, _ := tokenizer.Encode("Hello world")
text, _ := tokenizer.Decode([]int{123, 456})
```

### Scheduler (nano-vllm-go)

The Go scheduler handles continuous batching:

```go
// nanovllm/llm_engine.go
llm := nanovllm.NewLLMWithComponents(config, runner, tokenizer)
outputs, _ := llm.GenerateSimple(prompts, samplingParams, showProgress)
```

## Configuration

### Model Config (model_info.json)

Created automatically by export script:

```json
{
  "model_name": "Qwen/Qwen2-0.5B-Instruct",
  "vocab_size": 151936,
  "eos_token_id": 151643,
  "bos_token_id": 151643,
  "pad_token_id": 151643,
  "hidden_size": 896,
  "num_layers": 24,
  "num_heads": 14
}
```

### Sampling Parameters

Control generation behavior:

```go
samplingParams := nanovllm.NewSamplingParams(
    nanovllm.WithTemperature(0.7),  // Randomness (0.0 = deterministic, 1.0 = creative)
    nanovllm.WithMaxTokens(200),    // Max output length
    nanovllm.WithTopP(0.9),         // Nucleus sampling
    nanovllm.WithTopK(50),          // Top-K sampling
)
```

## Programmatic Usage

### Simple API

```go
package main

import (
    "fmt"
    "nano-vllm-go/nanovllm"
    "nano-vllm-go/purego"
)

func main() {
    // Setup
    config := nanovllm.NewConfig(".", nanovllm.WithEOS(151643))
    runner, _ := purego.NewONNXModelRunner("./models/qwen2-onnx/model.onnx", config)
    tokenizer, _ := purego.NewHFTokenizer("./models/qwen2-onnx")
    llm := nanovllm.NewLLMWithComponents(config, runner, tokenizer)
    defer llm.Close()

    // Generate
    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(100),
    )

    prompts := []string{"What is AI?"}
    outputs, _ := llm.GenerateSimple(prompts, samplingParams, false)

    fmt.Println(outputs[0].Text)
}
```

### Streaming API

```go
// Stream tokens as they're generated
for output := range llm.Generate(prompts, samplingParams) {
    if output.Err != nil {
        log.Fatal(output.Err)
    }
    fmt.Print(output.Text)  // Print each token
}
```

## Performance

### ONNX Runtime Performance

**CPU Inference:**
- Prefill: 50-100 tok/s (depends on prompt length)
- Decode: 20-40 tok/s (single token generation)

**With GPU (CUDA):**
- Prefill: 500-1000 tok/s
- Decode: 100-200 tok/s

### Optimization Tips

**1. Enable GPU (if available):**
```go
// In onnx_runner.go, enable CUDA provider
options.AppendExecutionProvider("CUDAExecutionProvider")
```

**2. Batch multiple requests:**
```go
// Process multiple prompts together
prompts := []string{"Q1?", "Q2?", "Q3?"}
outputs, _ := llm.GenerateSimple(prompts, samplingParams, true)
```

**3. Use smaller models:**
- Development: Qwen2-0.5B (600MB)
- Production: Qwen2-1.5B (3GB) or Qwen2-7B (14GB)

## Troubleshooting

### ONNX Runtime Not Found

```bash
# Install ONNX Runtime
# The Go package should auto-download, but if not:

# macOS
brew install onnxruntime

# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export LD_LIBRARY_PATH=$PWD/onnxruntime-linux-x64-1.16.0/lib:$LD_LIBRARY_PATH
```

### Model Export Fails

```bash
# Check Python dependencies
pip install --upgrade torch transformers onnx

# Use smaller model for testing
python3 scripts/export_to_onnx.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --output ./models/qwen2-onnx \
  --max-length 512
```

### Out of Memory

```bash
# Use a smaller model
python3 scripts/export_to_onnx.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --output ./models/qwen2-onnx

# Or reduce batch size in config
config := nanovllm.NewConfig(".",
    nanovllm.WithMaxNumSeqs(16),        # Reduce from 32
    nanovllm.WithMaxNumBatchedTokens(1024),  # Reduce from 2048
)
```

### Slow Inference

This is expected for CPU inference. For faster inference:

1. **Use GPU** (CUDA provider)
2. **Use smaller model** (Qwen2-0.5B vs 7B)
3. **Reduce max_tokens** in sampling params
4. **Enable quantization** (coming soon)

## Next Steps

### After Testing Works

**Deploy as single binary:**
```bash
# Build with static linking
go build -tags static -o dist/vllm-server ./cmd/server

# Deploy anywhere
./dist/vllm-server --model ./models/qwen2-onnx
```

**Add REST API:**
```go
// cmd/server/main.go
http.HandleFunc("/v1/completions", func(w http.ResponseWriter, r *http.Request) {
    // ... parse request
    outputs, _ := llm.GenerateSimple(prompts, samplingParams, false)
    json.NewEncoder(w).Encode(outputs)
})
```

**Add streaming:**
```go
// Stream server-sent events
w.Header().Set("Content-Type", "text/event-stream")
for output := range llm.Generate(prompts, samplingParams) {
    fmt.Fprintf(w, "data: %s\n\n", output.Text)
    w.(http.Flusher).Flush()
}
```

## Comparison with Other Approaches

### ONNX vs HTTP Backend

| Feature | ONNX | HTTP (Python) |
|---------|------|---------------|
| Setup time | 15-30 min | 5 min |
| Runtime deps | None | Python |
| Performance | 70-80% PyTorch | ~90% PyTorch |
| Deployment | Single binary | Two processes |
| Debugging | Harder | Easier |
| Best for | Production | Development |

### ONNX vs PyTorch (LibTorch)

| Feature | ONNX | PyTorch |
|---------|------|---------|
| Performance | 70-80% | 100% |
| Setup | Easier | Harder |
| Model support | Limited | Full |
| GPU support | Good | Excellent |
| Best for | Most cases | Max performance |

## Summary

**ONNX implementation gives you:**

✅ Real model inference
✅ Pure Go runtime
✅ Good performance
✅ Single binary deployment
✅ Cross-platform support

**Perfect for:**
- Production deployments
- Edge devices
- Containerized apps
- Cases where you can't use Python

**Start testing:**
```bash
# 1. Export model (one-time)
python3 scripts/export_to_onnx.py --model Qwen/Qwen2-0.5B-Instruct --output ./models/qwen2-onnx

# 2. Build
go build -o bin/onnx_test ./purego/example_onnx

# 3. Run
./bin/onnx_test "What is the capital of France?"
```

That's it! You're running real LLMs with nano-vllm-go! 🚀
