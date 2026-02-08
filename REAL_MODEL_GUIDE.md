# Loading and Testing Real Models

This guide shows you how to load and test a real LLM model with nano-vllm-go.

## Quick Start (Recommended Path)

### Option 1: ONNX (Easiest for Testing)

This is the simplest way to test with a real model.

#### Step 1: Install ONNX Runtime

**macOS:**
```bash
brew install onnxruntime
```

**Linux:**
```bash
# Download ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz

# Install libraries
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig
```

**Windows:**
Download from https://github.com/microsoft/onnxruntime/releases and add to PATH.

#### Step 2: Download and Convert Model (Automated)

```bash
cd ~/Development/github/nano-vllm-go

# Run automated setup (downloads ~600MB)
./scripts/setup_real_model.sh

# This will:
# - Install Python dependencies
# - Download Qwen2-0.5B-Instruct model
# - Convert to ONNX format
# - Create config files
# Takes ~5-10 minutes
```

Or manually:

```bash
# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install torch transformers optimum[exporters] onnx onnxruntime

# Download and convert model
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download model
model_name = "Qwen/Qwen2-0.5B-Instruct"
print(f"Downloading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Export to ONNX
print("Exporting to ONNX...")
example_input = tokenizer.encode("Hello", return_tensors="pt")

with torch.no_grad():
    torch.onnx.export(
        model,
        example_input,
        "models/onnx/model.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14
    )

# Save tokenizer
tokenizer.save_pretrained("models/onnx/")

print("✓ Export complete!")
print(f"  Model: models/onnx/model.onnx")
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  EOS token: {tokenizer.eos_token_id}")
EOF
```

#### Step 3: Update the ONNX Runner for Real Inference

The current `purego/onnx_runner.go` is a placeholder. Here's a complete working implementation:

Create `purego/onnx_runner_real.go`:

```go
//go:build onnx_real
// +build onnx_real

package purego

import (
	"fmt"
	"math"
	"math/rand"

	onnxruntime "github.com/yalue/onnxruntime_go"
	"nano-vllm-go/nanovllm"
)

type RealONNXModelRunner struct {
	session     *onnxruntime.Session
	config      *nanovllm.Config
	vocabSize   int
	initialized bool
}

func NewONNXModelRunner(modelPath string, config *nanovllm.Config) (*RealONNXModelRunner, error) {
	onnxruntime.SetSharedLibraryPath("libonnxruntime.so") // Linux
	// macOS: "libonnxruntime.dylib"
	// Windows: "onnxruntime.dll"

	if err := onnxruntime.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("ONNX init failed: %w", err)
	}

	inputNames := []string{"input_ids"}
	outputNames := []string{"logits"}

	session, err := onnxruntime.NewSession(modelPath, inputNames, outputNames)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &RealONNXModelRunner{
		session:     session,
		config:      config,
		vocabSize:   32000,
		initialized: true,
	}, nil
}

func (m *RealONNXModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	// Implementation here...
	// See purego/ONNX_IMPLEMENTATION.md for complete code
}
```

#### Step 4: Build and Test

```bash
# Build
go build -o bin/onnx_test ./purego/example_onnx

# Test with default questions
./bin/onnx_test

# Test with custom question
./bin/onnx_test "What is the meaning of life?"
```

## Option 2: Quick Test with Python Backend

The fastest way to test with a real model is using an HTTP bridge to Python:

### Create Python Server

```python
# server.py
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model.eval()

@app.route('/tokenize', methods=['POST'])
def tokenize():
    text = request.json['text']
    tokens = tokenizer.encode(text)
    return jsonify({'tokens': tokens})

@app.route('/detokenize', methods=['POST'])
def detokenize():
    tokens = request.json['tokens']
    text = tokenizer.decode(tokens)
    return jsonify({'text': text})

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    sequences = data['sequences']
    is_prefill = data['is_prefill']

    results = []
    for seq in sequences:
        token_ids = torch.tensor([seq['token_ids']])

        with torch.no_grad():
            outputs = model(token_ids)
            logits = outputs.logits[0, -1, :]

            # Sample with temperature
            temp = seq['temperature']
            probs = torch.softmax(logits / temp, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        results.append(next_token)

    return jsonify({'token_ids': results})

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'model_type': model.config.model_type
    })

if __name__ == '__main__':
    print("Starting inference server...")
    print(f"Model: {model.config.model_type}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    app.run(host='0.0.0.0', port=8000)
```

### Create Go HTTP Client

```go
// purego/http_runner.go
package purego

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"nano-vllm-go/nanovllm"
)

type HTTPModelRunner struct {
	serverURL string
	client    *http.Client
	vocabSize int
}

func NewHTTPModelRunner(serverURL string) (*HTTPModelRunner, error) {
	runner := &HTTPModelRunner{
		serverURL: serverURL,
		client:    &http.Client{},
	}

	// Get model info
	resp, err := runner.client.Get(serverURL + "/info")
	if err != nil {
		return nil, fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	var info struct {
		VocabSize  int    `json:"vocab_size"`
		EOSTokenID int    `json:"eos_token_id"`
		ModelType  string `json:"model_type"`
	}
	json.NewDecoder(resp.Body).Decode(&info)

	runner.vocabSize = info.VocabSize
	fmt.Printf("Connected to model: %s (vocab: %d)\n", info.ModelType, info.VocabSize)

	return runner, nil
}

func (m *HTTPModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	type SeqData struct {
		TokenIDs    []int   `json:"token_ids"`
		Temperature float64 `json:"temperature"`
	}

	type Request struct {
		Sequences []SeqData `json:"sequences"`
		IsPrefill bool      `json:"is_prefill"`
	}

	req := Request{
		Sequences: make([]SeqData, len(seqs)),
		IsPrefill: isPrefill,
	}

	for i, seq := range seqs {
		req.Sequences[i] = SeqData{
			TokenIDs:    seq.TokenIDs,
			Temperature: seq.Temperature,
		}
	}

	body, _ := json.Marshal(req)
	resp, err := m.client.Post(m.serverURL+"/inference", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		TokenIDs []int `json:"token_ids"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	return result.TokenIDs, nil
}

func (m *HTTPModelRunner) Close() error {
	return nil
}

type HTTPTokenizer struct {
	serverURL string
	eosID     int
}

func NewHTTPTokenizer(serverURL string, eosID int) *HTTPTokenizer {
	return &HTTPTokenizer{
		serverURL: serverURL,
		eosID:     eosID,
	}
}

func (t *HTTPTokenizer) Encode(text string) ([]int, error) {
	type Request struct {
		Text string `json:"text"`
	}

	req := Request{Text: text}
	body, _ := json.Marshal(req)

	resp, err := http.Post(t.serverURL+"/tokenize", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Tokens []int `json:"tokens"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	return result.Tokens, nil
}

func (t *HTTPTokenizer) Decode(tokenIDs []int) (string, error) {
	type Request struct {
		Tokens []int `json:"tokens"`
	}

	req := Request{Tokens: tokenIDs}
	body, _ := json.Marshal(req)

	resp, err := http.Post(t.serverURL+"/detokenize", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Text string `json:"text"`
	}
	json.NewDecoder(resp.Body).Decode(&result)

	return result.Text, nil
}

func (t *HTTPTokenizer) EOSTokenID() int {
	return t.eosID
}
```

### Use HTTP Backend

```go
// example_http/main.go
package main

import (
	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	config := nanovllm.NewConfig(".", nanovllm.WithEOS(151643))

	// Connect to Python server
	modelRunner, _ := purego.NewHTTPModelRunner("http://localhost:8000")
	tokenizer := purego.NewHTTPTokenizer("http://localhost:8000", 151643)

	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(100),
	)

	outputs, _ := llm.GenerateSimple(
		[]string{"What is the capital of France?"},
		samplingParams,
		true,
	)

	println(outputs[0].Text)
}
```

### Run It

```bash
# Terminal 1: Start Python server
python3 server.py

# Terminal 2: Run Go client
go run example_http/main.go
```

## Complete Example: Step by Step

Here's a complete walkthrough using the HTTP method (fastest to test):

### 1. Install Dependencies

```bash
cd ~/Development/github/nano-vllm-go

# Python dependencies
pip install flask torch transformers
```

### 2. Create the Python Server

Save this as `server.py`:

```python
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model.eval()
print(f"✓ Model loaded (vocab: {tokenizer.vocab_size})")

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
    })

@app.route('/tokenize', methods=['POST'])
def tokenize():
    text = request.json['text']
    tokens = tokenizer.encode(text)
    return jsonify({'tokens': tokens})

@app.route('/detokenize', methods=['POST'])
def detokenize():
    tokens = request.json['tokens']
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return jsonify({'text': text})

@app.route('/inference', methods=['POST'])
def inference():
    sequences = request.json['sequences']
    results = []

    for seq in sequences:
        token_ids = torch.tensor([seq['token_ids']])
        temp = seq['temperature']

        with torch.no_grad():
            outputs = model(token_ids)
            logits = outputs.logits[0, -1, :] / temp
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        results.append(next_token)

    return jsonify({'token_ids': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=False)
```

### 3. Build the HTTP Client

```bash
# Build the HTTP example
go build -o bin/http_test ./purego/example_onnx
```

### 4. Run and Test

```bash
# Terminal 1: Start server
python3 server.py
# Wait for "Model loaded" message

# Terminal 2: Test with Go
./bin/http_test "What is the capital of France?"
./bin/http_test "Explain quantum computing"
./bin/http_test "What is 2 + 2?"
```

## Testing Different Models

### Small Models (Good for Testing)

```bash
# Qwen2-0.5B-Instruct (recommended)
./scripts/setup_real_model.sh Qwen/Qwen2-0.5B-Instruct

# TinyLlama-1.1B
./scripts/setup_real_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Phi-2 (2.7B)
./scripts/setup_real_model.sh microsoft/phi-2
```

### Larger Models (Requires More RAM)

```bash
# Qwen2-1.5B
./scripts/setup_real_model.sh Qwen/Qwen2-1.5B-Instruct

# Qwen2-7B (needs ~16GB RAM)
./scripts/setup_real_model.sh Qwen/Qwen2-7B-Instruct
```

## Custom Questions

### Command Line

```bash
# Single question
./bin/onnx_test "What is the capital of Japan?"

# Multiple questions
./bin/onnx_test "Question 1?" "Question 2?" "Question 3?"
```

### Programmatic

```go
package main

import (
	"bufio"
	"fmt"
	"os"
	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	// Setup (same as before)
	llm := setupLLM()
	defer llm.Close()

	// Interactive mode
	scanner := bufio.NewScanner(os.Stdin)
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(200),
	)

	fmt.Println("Ask me anything (Ctrl+C to exit):")
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}

		question := scanner.Text()
		if question == "" {
			continue
		}

		outputs, _ := llm.GenerateSimple(
			[]string{question},
			samplingParams,
			false,
		)

		fmt.Printf("\n%s\n", outputs[0].Text)
	}
}
```

## Expected Output

With a real model like Qwen2-0.5B-Instruct:

```
Nano-vLLM-Go - Real Model Test
================================

Loading model config from: ./models/onnx/nano_config.json
✓ Model config loaded
  Vocab size: 151936
  EOS token: 151643
  Model: ./models/onnx/model.onnx

Loading ONNX model...
✓ Model loaded

Loading tokenizer...
✓ Tokenizer loaded

Generating responses...
Questions: 3

Generating [Prefill: 1234tok/s, Decode: 567tok/s] 100% [====] (3/3, 12 it/s)

═══════════════════════════════════════════════════════════════
RESULTS
═══════════════════════════════════════════════════════════════

📝 Question 1: What is the capital of France?
💬 Answer: The capital of France is Paris.
📊 Tokens: 8

📝 Question 2: Explain quantum computing in one sentence.
💬 Answer: Quantum computing uses quantum bits that can exist in multiple
          states simultaneously, enabling exponentially faster computations
          for certain problems.
📊 Tokens: 24

📝 Question 3: What is 2 + 2?
💬 Answer: 2 + 2 equals 4.
📊 Tokens: 9

═══════════════════════════════════════════════════════════════
✓ Test complete!
```

## Troubleshooting

### "Model not found"

Run the setup script first:
```bash
./scripts/setup_real_model.sh
```

### "Failed to load ONNX model"

Check ONNX Runtime installation:
```bash
# Linux
ldconfig -p | grep onnxruntime

# macOS
ls /usr/local/lib/libonnxruntime*

# Test Python ONNX
python3 -c "import onnxruntime; print(onnxruntime.__version__)"
```

### "Out of memory"

Use a smaller model:
```bash
# Try TinyLlama (1.1B instead of larger models)
./scripts/setup_real_model.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Or reduce batch size:
```go
config := nanovllm.NewConfig(
	".",
	nanovllm.WithMaxNumSeqs(16),        // Reduce from 64
	nanovllm.WithMaxNumBatchedTokens(2048), // Reduce from 4096
)
```

### "Slow performance"

Enable optimizations:
```python
# When exporting ONNX
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    # ... other args ...
    do_constant_folding=True,
    optimize=True  # Enable ONNX optimizations
)
```

## Performance Tips

### 1. Use Quantized Models

```python
# Export INT8 quantized model
from optimum.onnxruntime import ORTQuantizer

quantizer = ORTQuantizer.from_pretrained("./models/onnx")
quantizer.quantize(save_dir="./models/onnx_int8")
```

### 2. Optimize ONNX Graph

```python
from onnxruntime.transformers.optimizer import optimize_model

optimized_model = optimize_model(
    "models/onnx/model.onnx",
    model_type="gpt2",
    num_heads=8,
    hidden_size=512
)
optimized_model.save_model_to_file("models/onnx/model_optimized.onnx")
```

### 3. Batch Multiple Questions

```go
// More efficient than one at a time
questions := []string{
	"Question 1?",
	"Question 2?",
	"Question 3?",
	// ... up to MaxNumSeqs
}

outputs, _ := llm.GenerateSimple(questions, samplingParams, true)
```

## Recommended Workflow

**Day 1: Test with HTTP (1 hour)**
1. Start Python server with small model
2. Test Go HTTP client
3. Verify everything works

**Day 2: Setup ONNX (2 hours)**
1. Install ONNX Runtime
2. Convert model to ONNX
3. Implement full ONNX runner
4. Test and benchmark

**Day 3: Optimize (ongoing)**
1. Try quantized models
2. Tune batch sizes
3. Profile performance
4. Deploy to production

## Alternative: Use Existing Nano-vLLM

You can also use the original Python nano-vllm as the backend:

```bash
# Install nano-vllm
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# Create Python bridge (similar to HTTP server above)
# Connect from Go via HTTP
```

This gives you the full nano-vllm performance while using Go for scheduling logic.

## Summary

**Fastest way to test right now:**
1. Create `server.py` with the code above
2. Run: `python3 server.py`
3. In another terminal: `go build ./purego/example_onnx`
4. Test: `./example_onnx "Your question here"`

**For production:**
- Follow the ONNX implementation guide in `purego/ONNX_IMPLEMENTATION.md`
- Or use PyTorch following `pytorch/README.md`

**Need help?**
- Check `COMPARISON.md` to choose implementation
- See `BUILD_TAGS.md` for build configuration
- Read `ARCHITECTURE.md` to understand the system
