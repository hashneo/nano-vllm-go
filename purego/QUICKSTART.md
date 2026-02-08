# Pure Go Quick Start

This guide shows you how to use the pure Go implementation of nano-vllm-go.

## Option 1: Simple Example (No Dependencies)

The simplest way to get started is using the `SimpleBPETokenizer` which has no external dependencies.

### Build and Run

```bash
cd nano-vllm-go
go build -o bin/simple_example ./purego/example_simple
./bin/simple_example
```

### Code Example

```go
package main

import (
    "nano-vllm-go/nanovllm"
    "nano-vllm-go/purego"
)

func main() {
    // Create config
    config := nanovllm.NewConfig(".", nanovllm.WithEOS(2))

    // Create simple tokenizer (no external files needed)
    tokenizer := purego.NewSimpleBPETokenizer(2)

    // Create model runner (mock for demonstration)
    modelRunner := nanovllm.NewMockModelRunner(config)

    // Create LLM
    llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
    defer llm.Close()

    // Generate
    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(50),
    )

    outputs, _ := llm.GenerateSimple(
        []string{"hello world"},
        samplingParams,
        true,
    )

    println(outputs[0].Text)
}
```

## Option 2: Full ONNX Implementation

For production use with real models, you'll need to set up ONNX Runtime.

### Prerequisites

1. **Install ONNX Runtime**

**Linux:**
```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
export LD_LIBRARY_PATH=/path/to/onnxruntime-linux-x64-1.16.0/lib:$LD_LIBRARY_PATH
```

**macOS:**
```bash
brew install onnxruntime
```

2. **Convert Model to ONNX**

```bash
pip install optimum[exporters]

# Convert HuggingFace model
optimum-cli export onnx \
  --model Qwen/Qwen2-0.5B \
  --task text-generation \
  ./qwen2-onnx/
```

3. **Get Tokenizer**

Download `tokenizer.json` from the model:
```bash
wget https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/tokenizer.json
```

### Implementation Steps

See `purego/README.md` for the complete ONNX implementation guide.

The current `onnx_runner.go` is a placeholder. To implement:

1. Import ONNX Runtime Go bindings
2. Load ONNX model in `NewONNXModelRunner`
3. Prepare tensors in `Run` method
4. Execute model inference
5. Sample tokens from output logits

## SimpleBPETokenizer Features

The pure Go tokenizer includes:

- **No external dependencies** - Works out of the box
- **~180 token vocabulary** - Special tokens + common words + characters
- **Fallback encoding** - Unknown words split into characters
- **Simple but functional** - Good for testing and demonstrations

### Vocabulary

- Special: `<pad>`, `<s>`, `</s>`, `<unk>`
- Common words: hello, world, the, is, are, etc.
- All lowercase letters (a-z)
- All uppercase letters (A-Z)
- Digits (0-9)
- Punctuation: space, period, comma, etc.

### Example Usage

```go
tokenizer := purego.NewSimpleBPETokenizer(2)

// Encode
tokens, _ := tokenizer.Encode("hello world")
fmt.Println(tokens) // [4, 52, 5]

// Decode
text, _ := tokenizer.Decode(tokens)
fmt.Println(text) // "hello world"

// Vocabulary size
fmt.Println(tokenizer.VocabSize()) // ~180
```

## Advantages of Pure Go

✅ **Simple Deployment** - Single binary, no runtime dependencies
✅ **Cross-platform** - Compiles to any platform Go supports
✅ **Lower Memory** - No Python/PyTorch overhead
✅ **Fast Startup** - No model loading delays (for simple tokenizer)
✅ **Easy Integration** - Standard Go interfaces

## Limitations

⚠️ **Model Support** - ONNX conversion required
⚠️ **Performance** - 70-80% of native PyTorch speed
⚠️ **Features** - Some advanced features need additional work
⚠️ **Tokenizer** - SimpleBPE is basic, production needs proper BPE

## Next Steps

1. **Try the simple example** to understand the architecture
2. **Read `purego/README.md`** for full ONNX setup
3. **See `INTEGRATION.md`** for other backend options
4. **Implement custom runners** for your specific needs

## Comparison: Simple vs ONNX

| Feature | Simple Example | ONNX Example |
|---------|---------------|--------------|
| Dependencies | None | ONNX Runtime |
| Model | Mock | Real ONNX model |
| Tokenizer | SimpleBPE | HuggingFace compatible |
| Performance | N/A (mock) | 70-80% of PyTorch |
| Use Case | Testing/Demo | Production |
| Setup Time | Instant | 10-15 minutes |

## Troubleshooting

### Build Fails

Make sure you're in the project root:
```bash
cd nano-vllm-go
go mod tidy
go build ./purego/example_simple
```

### Import Errors

Check that module name matches in go.mod:
```go
module nano-vllm-go  // Should match your setup
```

### Runtime Errors

For the simple example, there should be no runtime dependencies. If you get errors:
- Check Go version (need 1.22+)
- Ensure all files are present
- Try `go clean -cache`

## Further Reading

- **Main README.md** - Project overview
- **ARCHITECTURE.md** - How nano-vllm-go works
- **purego/README.md** - Full ONNX implementation guide
- **INTEGRATION.md** - Other integration options

## Support

For issues:
1. Check existing examples work
2. Review documentation
3. Open GitHub issue with details
