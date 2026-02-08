# Complete ONNX Implementation Guide

This guide provides a complete, production-ready ONNX implementation for nano-vllm-go.

## Full Implementation

Replace the placeholder `onnx_runner.go` with this complete implementation:

```go
package purego

import (
	"fmt"
	"math"
	"math/rand"

	onnxruntime "github.com/yalue/onnxruntime_go"
	"nano-vllm-go/nanovllm"
)

type ONNXModelRunner struct {
	session     *onnxruntime.DynamicAdvancedSession
	config      *nanovllm.Config
	vocabSize   int
	initialized bool
}

func NewONNXModelRunner(modelPath string, config *nanovllm.Config) (*ONNXModelRunner, error) {
	// Set library path based on OS
	// Linux: "libonnxruntime.so"
	// macOS: "libonnxruntime.dylib"
	// Windows: "onnxruntime.dll"
	onnxruntime.SetSharedLibraryPath("libonnxruntime.so")

	if err := onnxruntime.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX: %w", err)
	}

	options, err := onnxruntime.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create options: %w", err)
	}
	defer options.Destroy()

	// CPU execution
	// For GPU: options.AppendExecutionProviderCUDA(0)

	session, err := onnxruntime.NewDynamicAdvancedSession(modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		options)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &ONNXModelRunner{
		session:     session,
		config:      config,
		vocabSize:   32000,
		initialized: true,
	}, nil
}

func (m *ONNXModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	batchSize := len(seqs)

	// Determine sequence length
	var maxLen int
	if isPrefill {
		for _, seq := range seqs {
			if seq.Len() > maxLen {
				maxLen = seq.Len()
			}
		}
	} else {
		maxLen = 1
	}

	// Prepare input tensors
	inputShape := onnxruntime.NewShape(int64(batchSize), int64(maxLen))
	inputData := make([]int64, batchSize*maxLen)
	maskData := make([]int64, batchSize*maxLen)

	for i, seq := range seqs {
		var tokens []int
		if isPrefill {
			tokens = seq.TokenIDs[:seq.Len()]
		} else {
			tokens = []int{seq.LastToken}
		}

		for j, token := range tokens {
			idx := i*maxLen + j
			inputData[idx] = int64(token)
			maskData[idx] = 1
		}
	}

	// Create tensors
	inputTensor, err := onnxruntime.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	maskTensor, err := onnxruntime.NewTensor(inputShape, maskData)
	if err != nil {
		return nil, fmt.Errorf("failed to create mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	// Run inference
	outputs, err := m.session.Run([]onnxruntime.ArbitraryTensor{
		{inputTensor, inputShape},
		{maskTensor, inputShape},
	})
	if err != nil {
		return nil, fmt.Errorf("inference failed: %w", err)
	}
	defer outputs[0].Tensor.Destroy()

	// Extract logits
	logitsData := outputs[0].Tensor.GetData()
	logits, ok := logitsData.([]float32)
	if !ok {
		return nil, fmt.Errorf("unexpected output type")
	}

	// Sample tokens
	tokenIDs := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		seqLen := seqs[i].Len()
		var startIdx int
		if isPrefill {
			startIdx = i*maxLen*m.vocabSize + (seqLen-1)*m.vocabSize
		} else {
			startIdx = i * m.vocabSize
		}

		temperature := seqs[i].Temperature
		tokenIDs[i] = m.sampleToken(logits[startIdx:startIdx+m.vocabSize], temperature)
	}

	return tokenIDs, nil
}

func (m *ONNXModelRunner) sampleToken(logits []float32, temperature float64) int {
	// Temperature scaling
	for i := range logits {
		logits[i] /= float32(temperature)
	}

	// Softmax
	maxLogit := float32(math.Inf(-1))
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float32
	probs := make([]float32, len(logits))
	for i, logit := range logits {
		probs[i] = float32(math.Exp(float64(logit - maxLogit)))
		sumExp += probs[i]
	}

	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample
	r := rand.Float32()
	var cumProb float32
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}

	return len(probs) - 1
}

func (m *ONNXModelRunner) Close() error {
	if m.session != nil {
		m.session.Destroy()
		m.session = nil
	}
	m.initialized = false
	return nil
}

func (m *ONNXModelRunner) SetVocabSize(size int) {
	m.vocabSize = size
}
```

## HuggingFace Tokenizer Implementation

For the tokenizer with actual HF tokenizers library:

```go
package purego

import (
	"fmt"
	"github.com/daulet/tokenizers"
)

type HFTokenizer struct {
	tokenizer *tokenizers.Tokenizer
	eosID     int
}

func NewHFTokenizer(tokenizerPath string) (*HFTokenizer, error) {
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &HFTokenizer{
		tokenizer: tk,
		eosID:     2,
	}, nil
}

func (t *HFTokenizer) Encode(text string) ([]int, error) {
	encoded, err := t.tokenizer.EncodeWithOptions(text, false, tokenizers.WithReturnTypeIDs())
	if err != nil {
		return nil, err
	}

	ids := encoded.IDs
	result := make([]int, len(ids))
	for i, id := range ids {
		result[i] = int(id)
	}

	return result, nil
}

func (t *HFTokenizer) Decode(tokenIDs []int) (string, error) {
	ids := make([]uint32, len(tokenIDs))
	for i, id := range tokenIDs {
		ids[i] = uint32(id)
	}

	text, err := t.tokenizer.Decode(ids, true)
	if err != nil {
		return "", err
	}

	return text, nil
}

func (t *HFTokenizer) EOSTokenID() int {
	return t.eosID
}

func (t *HFTokenizer) Close() error {
	if t.tokenizer != nil {
		t.tokenizer.Close()
		t.tokenizer = nil
	}
	return nil
}
```

## Complete Example

```go
package main

import (
	"fmt"
	"log"
	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	// Paths
	modelPath := "./model.onnx"
	tokenizerPath := "./tokenizer.json"

	// Config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(128),
		nanovllm.WithMaxNumBatchedTokens(8192),
		nanovllm.WithEOS(151643), // Qwen2 EOS token
	)

	// Load model and tokenizer
	modelRunner, err := purego.NewONNXModelRunner(modelPath, config)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer modelRunner.Close()

	tokenizer, err := purego.NewHFTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tokenizer.Close()

	// Update vocab size
	modelRunner.SetVocabSize(151936) // Qwen2-0.5B vocab size

	// Create LLM
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Generate
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(256),
	)

	prompts := []string{
		"What is the capital of France?",
		"Explain quantum computing in simple terms.",
	}

	fmt.Println("Generating...")
	outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	for i, output := range outputs {
		fmt.Printf("\nPrompt: %s\n", prompts[i])
		fmt.Printf("Output: %s\n", output.Text)
	}
}
```

## Model Conversion

### Using Optimum

```bash
# Install
pip install optimum[exporters] onnx

# Convert model
optimum-cli export onnx \
  --model Qwen/Qwen2-0.5B \
  --task text-generation-with-past \
  --optimize O2 \
  ./qwen2-onnx/

# This creates:
# - model.onnx (or decoder_model.onnx)
# - tokenizer.json
# - config.json
```

### Simplify Model

For better performance, export a simplified model:

```python
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Load model
model = ORTModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    export=True
)

# Save optimized
model.save_pretrained("./qwen2-onnx-optimized")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
tokenizer.save_pretrained("./qwen2-onnx-optimized")
```

## Performance Optimization

### 1. Graph Optimization

```go
// In NewONNXModelRunner
options.SetGraphOptimizationLevel(onnxruntime.AllOptimizations)
```

### 2. Thread Configuration

```go
options.SetIntraOpNumThreads(8)
options.SetInterOpNumThreads(8)
```

### 3. GPU Acceleration

```go
// Enable CUDA
err = options.AppendExecutionProviderCUDA(0)
if err != nil {
    // Fall back to CPU
    log.Printf("CUDA not available: %v", err)
}
```

### 4. Quantization

Export quantized model:

```bash
optimum-cli export onnx \
  --model Qwen/Qwen2-0.5B \
  --quantize arm64 \  # or avx512, avx2
  ./qwen2-onnx-quant/
```

## Testing

```bash
# Build
go build -o onnx_example ./purego/example

# Run with model
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
./onnx_example
```

## Troubleshooting

### Library Not Found

```bash
# Linux
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# macOS
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH

# Or copy to system location
sudo cp /path/to/libonnxruntime.* /usr/local/lib/
```

### Model Loading Fails

- Check ONNX version compatibility
- Verify model path is correct
- Ensure model is properly exported

### Slow Performance

- Enable optimization: `SetGraphOptimizationLevel`
- Use GPU if available
- Quantize model
- Increase batch size

### Out of Memory

- Reduce `MaxNumSeqs`
- Use quantized model
- Enable memory pattern optimization

## Production Checklist

- [ ] ONNX Runtime installed correctly
- [ ] Model converted and optimized
- [ ] Tokenizer compatible
- [ ] Correct vocab size configured
- [ ] EOS token ID correct
- [ ] Thread count optimized
- [ ] GPU enabled (if available)
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance tested

## References

- [ONNX Runtime](https://onnxruntime.ai/)
- [ONNX Runtime Go](https://github.com/yalue/onnxruntime_go)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum/)
- [daulet/tokenizers](https://github.com/daulet/tokenizers)
