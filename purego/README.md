# Pure Go Implementation

This directory contains pure Go implementations of model runners and tokenizers for nano-vllm-go, enabling deployment without C dependencies.

## Components

### 1. ONNX Model Runner (`onnx_runner.go`)

Uses `github.com/yalue/onnxruntime_go` to run ONNX models.

**Features:**
- Pure Go interface (though ONNX Runtime itself is C++)
- CPU and GPU support
- Standard ONNX model format
- Temperature-based sampling

**Limitations:**
- Requires ONNX Runtime shared library
- Slower than native PyTorch
- Single-batch inference (no KV cache optimization yet)

### 2. Tokenizers (`tokenizer.go`)

Two implementations provided:

#### HFTokenizer
Uses `github.com/daulet/tokenizers` (Rust tokenizers via CGo)
- Full HuggingFace tokenizer compatibility
- Fast tokenization
- Supports all tokenizer types (BPE, WordPiece, etc.)

#### SimpleBPETokenizer
Pure Go implementation for demonstration
- No external dependencies
- Very simple word-level tokenization
- Good for testing architecture
- Not suitable for production

## Installation

### Prerequisites

#### 1. ONNX Runtime

**Linux:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

**macOS:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-osx-arm64-1.16.0.tgz
tar xzf onnxruntime-osx-arm64-1.16.0.tgz
sudo cp onnxruntime-osx-arm64-1.16.0/lib/* /usr/local/lib/
```

**Windows:**
Download from https://github.com/microsoft/onnxruntime/releases and add to PATH.

#### 2. Go Dependencies

```bash
go get github.com/yalue/onnxruntime_go
go get github.com/daulet/tokenizers
```

## Converting Models to ONNX

### Using Optimum (Recommended)

```bash
pip install optimum[exporters]

# Convert a HuggingFace model
optimum-cli export onnx \
  --model Qwen/Qwen2-0.5B \
  --task text-generation \
  qwen2-onnx/
```

### Manual Conversion

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")
model.eval()

# Create dummy input
dummy_input = torch.randint(0, 32000, (1, 10))

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"}
    },
    opset_version=14
)
```

### Simplifying for Inference

For better performance, export only the LM head for decode steps:

```python
# Export optimized model for decoding
# (This requires custom model modifications)
```

## Usage

### Example 1: ONNX Model with HuggingFace Tokenizer

```go
package main

import (
    "nano-vllm-go/nanovllm"
    "nano-vllm-go/purego"
)

func main() {
    // Create config
    config := nanovllm.NewConfig(".", nanovllm.WithEOS(2))

    // Load ONNX model
    modelRunner, _ := purego.NewONNXModelRunner("model.onnx", config)
    defer modelRunner.Close()

    // Load tokenizer
    tokenizer, _ := purego.NewHFTokenizer("tokenizer.json")
    defer tokenizer.Close()

    // Create LLM
    llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
    defer llm.Close()

    // Generate
    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(100),
    )

    outputs, _ := llm.GenerateSimple(
        []string{"Hello, how are you?"},
        samplingParams,
        true,
    )

    println(outputs[0].Text)
}
```

### Example 2: Simple BPE for Testing

```go
package main

import (
    "nano-vllm-go/nanovllm"
    "nano-vllm-go/purego"
)

func main() {
    config := nanovllm.NewConfig(".", nanovllm.WithEOS(2))

    // Use simple tokenizer (no external files)
    tokenizer := purego.NewSimpleBPETokenizer(2)

    // Use mock model (no ONNX file needed)
    modelRunner := nanovllm.NewMockModelRunner(config)

    // Create LLM
    llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
    defer llm.Close()

    // Test the architecture
    samplingParams := nanovllm.NewSamplingParams()
    outputs, _ := llm.GenerateSimple([]string{"hello world"}, samplingParams, true)

    println(outputs[0].Text)
}
```

## Building

### Simple Example (No External Dependencies)

```bash
cd purego/example_simple
go build -o simple_example
./simple_example
```

### ONNX Example (Requires ONNX Runtime)

```bash
cd purego/example
go build -o onnx_example
./onnx_example
```

## Performance Tuning

### CPU Optimization

```go
// Set number of threads
options.SetIntraOpNumThreads(8)
options.SetInterOpNumThreads(8)
```

### GPU Acceleration

```go
// Enable CUDA (if available)
err = options.AppendExecutionProviderCUDA(0)
```

### Batch Processing

The ONNX runner supports batching:

```go
// Process multiple sequences in one inference call
// This is handled automatically by the scheduler
```

## Limitations and Trade-offs

### ONNX Runtime

**Pros:**
- Standard model format
- Good performance
- CPU and GPU support

**Cons:**
- Requires ONNX Runtime library (C++ dependency)
- Model conversion can be complex
- Limited KV cache optimization

### Pure Go vs Python

| Aspect | Pure Go | Python |
|--------|---------|--------|
| Dependencies | Fewer | Many (PyTorch, CUDA, etc.) |
| Deployment | Single binary | Full Python environment |
| Performance | 70-80% of native | 100% (native) |
| Development | More work | Easier (more libraries) |
| Memory usage | Lower | Higher |

## Troubleshooting

### "Failed to load ONNX model"

- Check that ONNX Runtime is installed
- Verify the model path is correct
- Ensure the model is compatible with ONNX Runtime version

### "Failed to load tokenizer"

- Verify tokenizer.json exists
- Check file permissions
- Ensure tokenizer format is compatible

### Slow Performance

- Enable GPU if available
- Increase batch size in config
- Use optimized ONNX models (quantization, fusion)
- Profile with Go pprof to find bottlenecks

### Out of Memory

- Reduce `MaxNumSeqs` and `MaxNumBatchedTokens`
- Use smaller models
- Enable quantization in ONNX export

## Advanced: Custom Model Runner

For maximum performance, implement a custom model runner:

```go
type CustomModelRunner struct {
    // Your optimized inference engine
}

func (m *CustomModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    // Your custom inference logic
    // Could use:
    // - Direct CUDA kernels
    // - Custom attention implementations
    // - Quantization
    // - KV cache sharing
}
```

## Future Improvements

- [ ] KV cache support in ONNX runner
- [ ] Flash Attention integration
- [ ] Quantization (INT8/INT4)
- [ ] Speculative decoding
- [ ] Better tokenizer performance
- [ ] Model fusion optimizations
- [ ] Distributed inference

## Resources

- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX Runtime Go](https://github.com/yalue/onnxruntime_go)
- [HuggingFace Tokenizers](https://github.com/daulet/tokenizers)
- [Optimum](https://huggingface.co/docs/optimum/index)
- [ONNX Model Zoo](https://github.com/onnx/models)
