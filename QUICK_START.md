# Quick Start Guide - Real Model Inference

Get up and running with real LLM models in nano-vllm-go!

## 🚀 Choose Your Mode

You have **3 options** for running real models. Pick based on your needs:

| Mode | Setup | Speed | Best For |
|------|-------|-------|----------|
| **HTTP** | 5 min | 40-80 tok/s | Development, quick testing |
| **ONNX** | 15 min | 20-40 tok/s | Production, single binary |
| **Native Go** | 5 min | 5-10 tok/s | Learning, no dependencies |

## 🔥 Option 1: HTTP Backend (Fastest Setup)

**What:** Go scheduler + Python server for inference
**Setup:** 5 minutes
**Speed:** 40-80 tokens/second

```bash
# One command does everything!
./scripts/start.sh http "What is the capital of France?"

# Or just:
./scripts/start_http.sh "Your question here"
```

**How it works:**
- Python server loads HuggingFace model (auto-downloads first time)
- Go client handles scheduling and batching
- Communicates via HTTP

**Use when:**
- You want to test quickly
- You're okay with Python dependency
- You're developing/experimenting

## ⚡ Option 2: ONNX Runtime (Production Ready)

**What:** Pure Go with ONNX model files
**Setup:** 15 minutes (one-time model export)
**Speed:** 20-40 tokens/second

```bash
# One command does everything!
./scripts/start.sh onnx "What is AI?"

# Or just:
./scripts/start_onnx.sh "Your question here"
```

**How it works:**
- Export HuggingFace model to ONNX format (one-time)
- Go code loads ONNX model directly
- Single binary deployment

**Use when:**
- You want production deployment
- You want single binary
- You don't want Python at runtime

## 🎓 Option 3: Pure Go Transformer (Educational)

**What:** Complete transformer implemented in Go from scratch
**Setup:** 5 minutes
**Speed:** 5-10 tokens/second

```bash
# One command does everything!
./scripts/start.sh native "Once upon a time"

# Or just:
./scripts/start_native.sh "Your prompt here"
```

**How it works:**
- Every operation (matmul, attention, etc.) in pure Go
- Loads GPT-2 weights
- Zero runtime dependencies

**Use when:**
- You want to learn how transformers work
- You want to modify the architecture
- You want zero dependencies
- Speed doesn't matter

## 📋 What Each Script Does

### start_http.sh
1. Checks/installs Python dependencies
2. Builds Go HTTP client
3. Starts Python server (if not running)
4. Runs your prompt
5. Keeps server running for more requests

### start_onnx.sh
1. Checks if ONNX model exists
2. If not: exports HuggingFace model to ONNX (one-time)
3. Builds Go ONNX runner
4. Runs your prompt

### start_native.sh
1. Checks if GPT-2 model downloaded
2. If not: downloads GPT-2 safetensors (one-time)
3. Builds Go native transformer
4. Runs your prompt

### start.sh (Master Script)
- Shows comparison table
- Dispatches to appropriate mode
- Provides help and examples

## 🎯 Examples

### Quick Test (HTTP - Fastest)
```bash
./scripts/start.sh http "What is machine learning?"
```

### Production Test (ONNX)
```bash
# First time: exports model (~10 min)
# Subsequent: instant start
./scripts/start.sh onnx "Explain quantum computing"
```

### Learn Transformers (Pure Go)
```bash
./scripts/start.sh native "The meaning of life is"
```

### Custom Models

**HTTP mode with different model:**
```bash
MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
  ./scripts/start_http.sh "Hello world"
```

**ONNX mode with different model:**
```bash
MODEL_NAME='microsoft/phi-2' \
  ./scripts/start_onnx.sh "Hello world"
```

### Multiple Questions

All modes support multiple prompts:
```bash
./scripts/start.sh http \
  "What is AI?" \
  "What is ML?" \
  "What is DL?"
```

## 🛠️ Manual Setup

If you prefer step-by-step control:

### HTTP Backend
```bash
# Terminal 1: Start server
python3 server.py

# Terminal 2: Run client
go build -o bin/http_test ./purego/example_http
./bin/http_test "Your question"
```

### ONNX Runtime
```bash
# One-time: Export model
python3 scripts/export_to_onnx.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --output ./models/qwen2-onnx

# Build and run
go build -o bin/onnx_test ./purego/example_onnx
MODEL_CONFIG=./models/qwen2-onnx/nano_config.json \
  ./bin/onnx_test "Your question"
```

### Pure Go Transformer
```bash
# One-time: Download GPT-2
python3 scripts/download_gpt2.py \
  --model gpt2 \
  --output ./models/gpt2

# Build and run
go build -o bin/native_test ./purego/example_native
MODEL_PATH=./models/gpt2/model.safetensors \
TOKENIZER_PATH=./models/gpt2 \
  ./bin/native_test "Your prompt"
```

## 📊 Performance Comparison

### HTTP Backend
```
┌──────────────┬─────────────┬─────────────┐
│ Metric       │ Value       │ Notes       │
├──────────────┼─────────────┼─────────────┤
│ Setup        │ 5 min       │ First time  │
│ Prefill      │ 100 tok/s   │ Depends     │
│ Decode       │ 40-80 tok/s │ on model    │
│ Memory       │ 600MB-2GB   │             │
│ Dependencies │ Python      │             │
└──────────────┴─────────────┴─────────────┘
```

### ONNX Runtime
```
┌──────────────┬─────────────┬─────────────┐
│ Metric       │ Value       │ Notes       │
├──────────────┼─────────────┼─────────────┤
│ Setup        │ 15 min      │ First time  │
│ Prefill      │ 50 tok/s    │ CPU only    │
│ Decode       │ 20-40 tok/s │             │
│ Memory       │ 600MB-2GB   │             │
│ Dependencies │ None        │ Runtime     │
└──────────────┴─────────────┴─────────────┘
```

### Pure Go
```
┌──────────────┬─────────────┬─────────────┐
│ Metric       │ Value       │ Notes       │
├──────────────┼─────────────┼─────────────┤
│ Setup        │ 5 min       │ First time  │
│ Prefill      │ 10-20 tok/s │ Educational │
│ Decode       │ 5-10 tok/s  │             │
│ Memory       │ 500MB       │ GPT-2 small │
│ Dependencies │ None        │ Zero!       │
└──────────────┴─────────────┴─────────────┘
```

## 🎨 Architecture Overview

### HTTP Mode
```
┌─────────────┐         ┌──────────────┐
│ Go Client   │  HTTP   │ Python Server│
│             │ <-----> │              │
│ • Scheduler │         │ • PyTorch    │
│ • Batching  │         │ • HF Models  │
└─────────────┘         └──────────────┘
```

### ONNX Mode
```
┌──────────────────────────┐
│   Single Go Process      │
│                          │
│ ┌──────────────────────┐ │
│ │ nano-vllm Scheduler  │ │
│ └──────────┬───────────┘ │
│            │             │
│ ┌──────────┴───────────┐ │
│ │ ONNX Runtime (CGo)   │ │
│ └──────────────────────┘ │
└──────────────────────────┘
```

### Pure Go Mode
```
┌──────────────────────────┐
│   Single Go Process      │
│                          │
│ ┌──────────────────────┐ │
│ │ nano-vllm Scheduler  │ │
│ └──────────┬───────────┘ │
│            │             │
│ ┌──────────┴───────────┐ │
│ │ Pure Go Transformer  │ │
│ │ • MatMul             │ │
│ │ • Attention          │ │
│ │ • All ops in Go!     │ │
│ └──────────────────────┘ │
└──────────────────────────┘
```

## 🐛 Troubleshooting

### HTTP Mode

**Server won't start:**
```bash
# Check Python version
python3 --version  # Need 3.8+

# Install dependencies
pip3 install flask torch transformers

# Check logs
tail -f /tmp/nano-vllm-server.log
```

**Connection refused:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Restart server
kill $(cat /tmp/nano-vllm-server.pid)
./scripts/start_http.sh
```

### ONNX Mode

**Model export fails:**
```bash
# Try smaller model
MODEL_NAME='Qwen/Qwen2-0.5B-Instruct' ./scripts/start_onnx.sh

# Check Python packages
pip3 install --upgrade torch transformers onnx
```

**ONNX Runtime errors:**
```bash
# The Go package should auto-download ONNX Runtime
# If it fails, check: https://onnxruntime.ai/

# On macOS:
brew install onnxruntime
```

### Pure Go Mode

**Download fails:**
```bash
# Manual download
python3 scripts/download_gpt2.py \
  --model gpt2 \
  --output ./models/gpt2

# Try smaller model if out of memory
# (gpt2 is already the smallest at 124M params)
```

## 📚 Learn More

- **HTTP Backend**: [TEST_REAL_MODEL.md](TEST_REAL_MODEL.md)
- **ONNX Runtime**: [ONNX_GUIDE.md](ONNX_GUIDE.md)
- **Pure Go**: [NATIVE_TRANSFORMER_GUIDE.md](NATIVE_TRANSFORMER_GUIDE.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [README.md](README.md)

## 💡 Recommendations

**Starting out?** → Use HTTP mode (fastest setup)
```bash
./scripts/start.sh http "Hello world"
```

**Going to production?** → Use ONNX mode (single binary)
```bash
./scripts/start.sh onnx "Hello world"
```

**Want to learn?** → Use Pure Go mode (see all the code)
```bash
./scripts/start.sh native "Hello world"
```

**Not sure?** → Try all three!
```bash
./scripts/start.sh http "What is AI?"
./scripts/start.sh onnx "What is AI?"
./scripts/start.sh native "What is AI?"
```

## 🎉 Summary

**You have 3 ways to run real LLM models:**

1. **HTTP** - Python server + Go client (fastest setup)
2. **ONNX** - Pure Go with ONNX models (production ready)
3. **Native** - Pure Go transformer (educational)

**All modes:**
- ✅ Use nano-vllm-go scheduler
- ✅ Support continuous batching
- ✅ Handle multiple prompts
- ✅ Work with real models
- ✅ Have one-command startup scripts

**Choose based on:**
- Speed requirements
- Deployment constraints
- Learning goals

**Get started now:**
```bash
./scripts/start.sh
```

Happy inferencing! 🚀
