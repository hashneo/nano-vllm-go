# Nano-vLLM-Go Quickstart

Get started with nano-vllm-go in 5 minutes!

## Instant Demo (0 minutes setup)

```bash
cd ~/Development/github/nano-vllm-go
make run-purego
```

**What you'll see:**
- Full nano-vllm architecture in action
- Continuous batching
- Scheduler working (prefill/decode)
- Tokenization (SimpleBPE)
- Progress bar with throughput
- Generated text output

**Output example:**
```
Nano-vLLM-Go - Simple BPE Example
===================================

Vocabulary size: 125

Testing tokenization:
  Input: hello world this is a test
  Tokens: [4 115 5 115 21 115 7 115 53 115 72 57 71 72]
  Decoded: hello world this is a test

Generating responses...
[████████████████] 100% (3/3, 17316 it/s)

Results:
========
Prompt 1: hello world
Output: ...generated text...
```

## Three Implementation Options

### 1. SimpleBPE (What you just ran)

**Use for:** Testing, learning, demos

**Features:**
- ✅ Zero dependencies
- ✅ 3MB binary
- ✅ Works immediately
- ✅ Mock model

**Build:**
```bash
go build -o bin/simple ./purego/example_simple
./bin/simple
```

### 2. ONNX (Production - CPU)

**Use for:** Production deployment, CPU inference

**Features:**
- ✅ Real models (70-80% PyTorch speed)
- ✅ ONNX format (standard)
- ✅ Good performance
- ⚠️ Requires ONNX Runtime

**Setup (15 minutes):**
```bash
# 1. Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/

# 2. Convert model
pip install optimum
optimum-cli export onnx --model Qwen/Qwen2-0.5B ./model-onnx

# 3. Build and run
go build -tags purego -o bin/onnx ./purego/example
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./bin/onnx
```

**See:** `purego/QUICKSTART.md` for detailed setup

### 3. PyTorch (Production - GPU)

**Use for:** Maximum performance, GPU acceleration

**Features:**
- ✅ Native PyTorch (100% speed)
- ✅ Full CUDA support
- ✅ Latest features
- ⚠️ Requires LibTorch setup

**Setup (60 minutes):**
```bash
# 1. Automated setup
./scripts/setup_pytorch.sh

# 2. Export model
python3 scripts/export_model.py --model Qwen/Qwen2-0.5B --output model.pt

# 3. Build and run
go build -tags pytorch -o bin/pytorch ./pytorch/example
export LD_LIBRARY_PATH=./third_party/libtorch/lib:$LD_LIBRARY_PATH
./bin/pytorch
```

**See:** `pytorch/README.md` for detailed setup

## Choosing Implementation

| Your Need | Best Choice |
|-----------|-------------|
| Learning/Testing | SimpleBPE |
| Production (CPU) | ONNX |
| Production (GPU) | PyTorch |
| Docker Deploy | ONNX |
| Max Performance | PyTorch |
| Zero Setup | SimpleBPE |

## Quick Commands Reference

```bash
# SimpleBPE (works now)
make run-purego

# ONNX (after setup)
go build -tags purego ./purego/example

# PyTorch (after setup)
go build -tags pytorch ./pytorch/example

# Run tests
go test ./nanovllm

# Build all
make build-all

# Clean
make clean
```

## What's Under the Hood

Even SimpleBPE demonstrates:
1. **Continuous Batching** - Multiple sequences processed dynamically
2. **Scheduler** - Prefill and decode phase separation
3. **Block Manager** - Memory allocation with prefix caching
4. **Sequence Management** - State tracking and completion
5. **Tokenization** - Full encode/decode pipeline

## Next Steps

### If you want to learn (15 minutes):
1. Run SimpleBPE example ✓ (you did this!)
2. Read `ARCHITECTURE.md` - Understand how it works
3. Modify `purego/tokenizer.go` - Add vocabulary
4. Experiment with batch sizes and temperatures

### If you want production (30-60 minutes):
1. Choose ONNX or PyTorch based on GPU availability
2. Follow setup guide (`purego/QUICKSTART.md` or `pytorch/README.md`)
3. Test with small model (Qwen2-0.5B)
4. Benchmark on your hardware
5. Deploy!

### If you want to compare (5 minutes):
1. Read `COMPARISON.md` - Detailed comparison
2. Check `BUILD_TAGS.md` - Build system explained
3. Decide which path to take

## Troubleshooting

### SimpleBPE not working?

```bash
# Verify build
cd ~/Development/github/nano-vllm-go
go build -o bin/test ./purego/example_simple
./bin/test

# Should output immediately
```

### Want to see the code?

```bash
# Core library
ls -la nanovllm/

# SimpleBPE tokenizer
cat purego/tokenizer.go | less

# Example
cat purego/example_simple/main.go
```

### Need help?

1. Check documentation:
   - `README.md` - Overview
   - `ARCHITECTURE.md` - Deep dive
   - `COMPARISON.md` - Choose implementation
   - `BUILD_TAGS.md` - Build system

2. Look at examples:
   - `purego/example_simple/` - SimpleBPE
   - `purego/example/` - ONNX
   - `pytorch/example/` - PyTorch

3. Review tests:
   - `nanovllm/*_test.go`

## Key Documentation

| Document | Purpose |
|----------|---------|
| **README.md** | Project overview |
| **QUICKSTART.md** | This file - get started fast |
| **ARCHITECTURE.md** | How nano-vllm works |
| **COMPARISON.md** | Compare implementations |
| **BUILD_TAGS.md** | Build system guide |
| **purego/QUICKSTART.md** | Pure Go setup |
| **pytorch/README.md** | PyTorch setup |

## Performance Expectations

### SimpleBPE (Mock Model)
- Purpose: Testing architecture
- Speed: N/A (mock)
- Memory: ~10MB

### ONNX (Real Model - Qwen2-0.5B)
- CPU: 50-80 tokens/sec
- GPU (T4): 200-300 tokens/sec
- Memory: ~500MB

### PyTorch (Real Model - Qwen2-0.5B)
- CPU: 60-100 tokens/sec
- GPU (T4): 300-500 tokens/sec
- GPU (A100): 1000-2000 tokens/sec
- Memory: ~1GB

## Summary

You've successfully:
1. ✅ Seen nano-vllm-go in action
2. ✅ Learned about three implementations
3. ✅ Know where to go next

**Choose your path:**
- **Stay with SimpleBPE** - Keep learning
- **Move to ONNX** - Production CPU
- **Jump to PyTorch** - Production GPU

**Ready to dive deeper?**
```bash
# Read architecture
cat ARCHITECTURE.md | less

# Compare implementations
cat COMPARISON.md | less

# Or just keep experimenting!
```

Happy coding! 🚀
