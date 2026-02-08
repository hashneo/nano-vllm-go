# Add this to the main README.md after the Quick Start section:

## 🎯 Choose Your Implementation

Nano-vLLM-Go supports three implementations via Go build tags:

| Implementation | Command | Setup | Use Case |
|----------------|---------|-------|----------|
| **SimpleBPE** | `make run-purego` | 0 min | Testing, learning |
| **ONNX** | `go build -tags purego` | 15 min | Production (CPU) |
| **PyTorch** | `go build -tags pytorch` | 60 min | Production (GPU) |

See [COMPARISON.md](COMPARISON.md) for detailed comparison and [BUILD_TAGS.md](BUILD_TAGS.md) for usage guide.

### Quick Commands

```bash
# Pure Go (SimpleBPE) - Works immediately
make run-purego

# ONNX - Requires ONNX Runtime
go build -tags purego -o bin/onnx ./purego/example

# PyTorch - Requires LibTorch setup
./scripts/setup_pytorch.sh
go build -tags pytorch -o bin/pytorch ./pytorch/example
```
