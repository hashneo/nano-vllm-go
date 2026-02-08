# Testing with a Real Model - Quick Guide

This guide shows you the **fastest way** to test nano-vllm-go with a real LLM model and ask actual questions.

## 🚀 Fastest Method: HTTP Backend (5 minutes)

This method uses Python for model inference and Go for scheduling. No ONNX conversion needed!

### Step 1: Install Python Dependencies (1 minute)

```bash
cd ~/Development/github/nano-vllm-go

pip install flask torch transformers
```

### Step 2: Start the Python Server (2 minutes)

```bash
# Start server (downloads model on first run ~600MB)
python3 server.py

# Or specify a different model
python3 server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

**Wait for:** "Ready to accept requests!" message

### Step 3: Test with Go (1 minute)

In a new terminal:

```bash
cd ~/Development/github/nano-vllm-go

# Build HTTP client
go build -o bin/http_test ./purego/example_http

# Test with default questions
./bin/http_test

# Test with your own question
./bin/http_test "What is the capital of France?"

# Multiple questions
./bin/http_test "Question 1?" "Question 2?" "Question 3?"
```

### Expected Output

```
Nano-vLLM-Go - HTTP Backend Test
==================================

Connecting to server: http://localhost:8000
✓ Connected to model (vocab: 151936)

Processing 1 question(s)...

Generating [Prefill: 234tok/s, Decode: 45tok/s] 100% [====] (1/1, 2 it/s)

═══════════════════════════════════════════════════════════════════
RESULTS
═══════════════════════════════════════════════════════════════════

📝 Question 1: What is the capital of France?
💬 Answer: The capital of France is Paris.
📊 Tokens: 8

═══════════════════════════════════════════════════════════════════
✓ Test complete!
```

## ✅ That's It!

You're now running **real model inference** with nano-vllm-go's scheduler!

The Go code handles:
- ✅ Continuous batching
- ✅ Sequence scheduling
- ✅ Memory management
- ✅ Progress tracking

Python handles:
- ✅ Model inference
- ✅ Tokenization
- ✅ Sampling

## 🎯 Testing Different Models

### Small Models (Good for Testing)

```bash
# Qwen2-0.5B-Instruct (recommended, ~600MB)
python3 server.py --model Qwen/Qwen2-0.5B-Instruct

# TinyLlama-1.1B (~2GB)
python3 server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Phi-2 (2.7B, ~5GB)
python3 server.py --model microsoft/phi-2
```

### Testing Tips

```bash
# Simple question
./bin/http_test "Hello, how are you?"

# Math question
./bin/http_test "What is 15 * 23?"

# Reasoning question
./bin/http_test "Why is the sky blue?"

# Multiple questions at once (tests batching)
./bin/http_test "What is AI?" "What is ML?" "What is DL?"
```

## 🔧 Advanced: Different Sampling Parameters

Create a custom test program:

```go
// test_custom.go
package main

import (
	"fmt"
	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	// Setup
	modelRunner, _ := purego.NewHTTPModelRunner("http://localhost:8000")
	defer modelRunner.Close()

	tokenizer := purego.NewHTTPTokenizer("http://localhost:8000", 151643)
	config := nanovllm.NewConfig(".", nanovllm.WithEOS(151643))
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Test different temperatures
	temperatures := []float64{0.3, 0.7, 1.0}
	question := "Write a creative story opening."

	for _, temp := range temperatures {
		fmt.Printf("\n🌡️  Temperature: %.1f\n", temp)

		samplingParams := nanovllm.NewSamplingParams(
			nanovllm.WithTemperature(temp),
			nanovllm.WithMaxTokens(50),
		)

		outputs, _ := llm.GenerateSimple([]string{question}, samplingParams, false)
		fmt.Printf("📝 Output: %s\n", outputs[0].Text)
	}
}
```

## 📊 What You're Testing

When you run the HTTP example, you're testing:

### Go Components (Your Code)
- ✅ Continuous batching scheduler
- ✅ Sequence management
- ✅ Block-based memory allocation
- ✅ Prefix caching logic
- ✅ Progress tracking

### Python Components (via HTTP)
- ✅ Real model inference (PyTorch)
- ✅ Proper tokenization (HuggingFace)
- ✅ Accurate text generation

### Architecture Benefits
- ✅ Go's scheduling efficiency
- ✅ Python's ML ecosystem
- ✅ Clean separation of concerns
- ✅ Easy to debug

## 🐛 Troubleshooting

### Server won't start

```bash
# Check Python version (need 3.8+)
python3 --version

# Check dependencies
pip list | grep -E "torch|transformers|flask"

# Reinstall if needed
pip install --upgrade torch transformers flask
```

### Connection refused

```bash
# Check server is running
curl http://localhost:8000/health

# Should return: {"status":"healthy","model_loaded":true}
```

### Out of memory

Use a smaller model:
```bash
# Try TinyLlama (smaller than Qwen2)
python3 server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Slow generation

This is expected for CPU inference. For GPU:

```python
# In server.py, change device_map
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use FP16 for GPU
    device_map="auto"  # Auto GPU
)
```

## 🎓 Next Steps

### After HTTP Testing Works

**Option A: Move to ONNX (Better Performance)**
```bash
# Convert model to ONNX
./scripts/setup_real_model.sh

# Implement full ONNX runner
# See purego/ONNX_IMPLEMENTATION.md
```

**Option B: Move to PyTorch (Maximum Performance)**
```bash
# Setup LibTorch
./scripts/setup_pytorch.sh

# Build PyTorch version
go build -tags pytorch ./pytorch/example
```

### Improve HTTP Backend

**Add caching:**
```python
# Cache tokenization results
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_tokenize(text):
    return tokenizer.encode(text)
```

**Add batching:**
```python
# Process multiple sequences at once
def batch_inference(sequences):
    all_token_ids = [seq['token_ids'] for seq in sequences]
    input_tensor = torch.tensor(all_token_ids)
    # ... run batched inference
```

## 📈 Performance Comparison

### HTTP Backend

**Pros:**
- ✅ Fast setup (5 minutes)
- ✅ Easy debugging
- ✅ No compilation needed
- ✅ Can update model without rebuilding Go

**Cons:**
- ❌ Network overhead (~1-5ms per request)
- ❌ Serialization overhead
- ❌ Need two processes

**Performance:**
- Local: ~90% of direct PyTorch
- Network overhead: 1-5ms
- Good for: Testing, development, distributed setups

### Direct ONNX/PyTorch

**Pros:**
- ✅ No network overhead
- ✅ Single process
- ✅ Better latency

**Cons:**
- ❌ More complex setup
- ❌ Harder to debug

**Performance:**
- ONNX: ~70-80% of PyTorch
- PyTorch: 100% (native)

## 🎯 Recommended Path

**For Testing RIGHT NOW:**
1. ✅ Use HTTP backend (this guide)
2. Test with questions
3. Verify everything works

**For Development:**
1. Keep using HTTP backend
2. Iterate on Go scheduler
3. Easy to debug both sides

**For Production:**
1. Move to ONNX or PyTorch
2. Optimize for your hardware
3. Deploy as single binary

## Complete Code Example

Here's a complete, runnable example you can try right now:

### server.py (Already created in project root)

```bash
# Start server
python3 server.py
```

### test_question.go

```go
package main

import (
	"fmt"
	"os"
	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: ./test_question \"Your question here\"")
		os.Exit(1)
	}

	question := os.Args[1]

	// Connect to server
	modelRunner, err := purego.NewHTTPModelRunner("http://localhost:8000")
	if err != nil {
		fmt.Printf("❌ Can't connect to server: %v\n", err)
		fmt.Println("\nStart the server first:")
		fmt.Println("  python3 server.py")
		os.Exit(1)
	}
	defer modelRunner.Close()

	tokenizer := purego.NewHTTPTokenizer("http://localhost:8000", 151643)
	config := nanovllm.NewConfig(".", nanovllm.WithEOS(151643))
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Generate
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(200),
	)

	fmt.Printf("\n📝 Question: %s\n\n", question)
	fmt.Println("Thinking...")

	outputs, _ := llm.GenerateSimple([]string{question}, samplingParams, false)

	fmt.Printf("\n💬 Answer: %s\n\n", outputs[0].Text)
	fmt.Printf("📊 Generated %d tokens\n", len(outputs[0].TokenIDs))
}
```

### Run it

```bash
# Build
go build -o test_question test_question.go

# Test
./test_question "What is the capital of France?"
./test_question "Explain quantum physics simply"
./test_question "Write a haiku about programming"
```

## 🎉 Summary

**To test with a real model RIGHT NOW:**

```bash
# Terminal 1: Start Python server
python3 server.py

# Terminal 2: Test with Go
go build -o bin/http_test ./purego/example_http
./bin/http_test "Your question here"
```

**That's it!** You're now using a real LLM model with nano-vllm-go's scheduling architecture.

The HTTP method gives you:
- ✅ Real model inference
- ✅ Proper tokenization
- ✅ Continuous batching
- ✅ All nano-vllm features
- ✅ Easy to test and debug

Once this works, you can move to ONNX or PyTorch for better performance in production.
