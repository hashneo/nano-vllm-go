# Test Real Model in 5 Minutes

The FASTEST way to test nano-vllm-go with a real model.

## Prerequisites

- Python 3.8+
- pip

## Steps

### 1. Install Python Dependencies (1 min)

```bash
pip install flask torch transformers
```

### 2. Start Server (2 min - downloads model)

```bash
cd ~/Development/github/nano-vllm-go
python3 server.py
```

Wait for: "Ready to accept requests!"

### 3. Build and Test (2 min)

**New terminal:**

```bash
cd ~/Development/github/nano-vllm-go

# Build
go build -o bin/http_test ./purego/example_http

# Test
./bin/http_test "What is the capital of France?"
```

## Done! 🎉

You're now running a real LLM model with nano-vllm-go.

## Try More Questions

```bash
./bin/http_test "Explain quantum computing"
./bin/http_test "What is 15 * 23?"
./bin/http_test "Write a haiku about Go programming"
```

## Multiple Questions at Once

```bash
./bin/http_test "What is AI?" "What is ML?" "What is DL?"
```

This tests the **continuous batching** feature!

## What's Happening

**Python Server:**
- Loads Qwen2-0.5B-Instruct model
- Handles tokenization and inference
- Returns next tokens

**Go Client:**
- Manages multiple sequences
- Schedules prefill/decode phases
- Handles continuous batching
- Tracks progress

## Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   Go Process    │         │  Python Server   │
│                 │         │                  │
│  • Scheduler    │  HTTP   │  • Model         │
│  • Batching     │ <─────> │  • Tokenizer     │
│  • Memory Mgmt  │         │  • Inference     │
└─────────────────┘         └──────────────────┘
```

## Next Steps

Once this works:

**Better Performance:**
- Use ONNX (70-80% PyTorch speed, single process)
- Use PyTorch directly (100% speed, native)

**See:**
- `REAL_MODEL_GUIDE.md` - Complete guide
- `purego/QUICKSTART.md` - ONNX setup
- `pytorch/README.md` - PyTorch setup
- `COMPARISON.md` - Choose implementation

## Troubleshooting

**"Connection refused"**
```bash
# Check server is running
curl http://localhost:8000/health
```

**"Model not found"**
```bash
# Server will auto-download on first run
# Or specify local model:
python3 server.py --model /path/to/local/model
```

**Slow on first run**
```bash
# First run downloads ~600MB model
# Subsequent runs are instant
```

## Summary

✅ **Working:** Real model inference via HTTP
✅ **Fast:** 5 minutes to test
✅ **Easy:** No complex setup
✅ **Flexible:** Change models easily

**Now running:**
- Real Qwen2-0.5B-Instruct model
- Nano-vllm-go scheduling
- Continuous batching
- Full architecture demo

Ask your questions! 🚀
