# GPT-2 Quick Start Guide

## Running Questions Through GPT-2

### Option 1: Use the provided script

```bash
# Default question
go run ask_gpt2.go

# Custom question
go run ask_gpt2.go "Once upon a time"

# Or build and run
go build ask_gpt2.go
./ask_gpt2 "The meaning of life is"
```

### Option 2: Write your own script

```go
package main

import (
	"fmt"
	"log"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	modelDir := "./models/gpt2-small"

	// Create model runner
	modelRunner, _ := nanovllm.NewTensorModelRunner(modelDir)

	// Create tokenizer
	tokenizer, _ := purego.NewGPT2Tokenizer(modelDir)

	// Create config
	config := nanovllm.NewConfig(
		modelDir,
		nanovllm.WithMaxNumSeqs(1),
		nanovllm.WithMaxModelLen(512),
	)

	// Create LLM
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set generation parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithMaxTokens(20),
		nanovllm.WithTemperature(0.7),
	)

	// Generate
	outputs, _ := llm.GenerateSimple([]string{"Hello world"}, samplingParams, true)
	fmt.Println(outputs[0].Text)
}
```

## Important Notes

### Performance Warning ⚠️
The current implementation is **slow** because it lacks KV caching. Each token generation requires reprocessing the entire sequence:
- Token 1: Process 10 tokens
- Token 2: Process 11 tokens (recomputes first 10)
- Token 3: Process 12 tokens (recomputes first 11)
- etc.

This is O(N²) complexity. For 20 tokens, it does ~200 forward passes instead of 20.

### Tokenizer Limitation
The GPT-2 tokenizer uses simplified word-level tokenization, not proper BPE (Byte Pair Encoding). This means:
- Output quality will be lower than full GPT-2
- Some words may not tokenize correctly
- For production use, integrate a proper BPE tokenizer

### What Works
✅ Model loads and runs correctly
✅ Logits have proper variation
✅ Basic text generation works
✅ Greedy sampling (picks highest probability token)

### What's Missing
❌ KV caching (causes slow generation)
❌ Advanced sampling (temperature/top-p/top-k not yet implemented)
❌ Proper BPE tokenizer
❌ Batch processing

## Troubleshooting

**Error: "no such file or directory"**
- Make sure GPT-2 model is downloaded to `./models/gpt2-small`
- Run from the repo root directory

**Very slow generation**
- This is expected without KV caching
- 20 tokens takes ~10-20 seconds on CPU
- Consider reducing `WithMaxTokens()` to 10 for faster testing

**Garbage output**
- The simplified tokenizer may produce odd results
- Try simpler prompts like "Hello world" or "The cat"
