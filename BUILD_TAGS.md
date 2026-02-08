# Build Tags Guide

This document explains how to use Go build tags to switch between different implementations of nano-vllm-go.

## Overview

Nano-vLLM-Go supports multiple backends through Go build tags:

- **Pure Go (default)** - ONNX Runtime, no C dependencies (except ONNX)
- **PyTorch** - Native PyTorch via LibTorch C++ API (maximum performance)

## Quick Start

### Build with Default (Pure Go)

```bash
# Standard build uses Pure Go ONNX implementation
go build ./example
./example

# Or explicitly specify
go build -tags purego ./purego/example_simple
```

### Build with PyTorch

```bash
# Build with PyTorch backend
go build -tags pytorch ./pytorch/example
./pytorch_example
```

## Available Build Tags

### `purego` (Default)

**Description:** Pure Go implementation using ONNX Runtime

**When to use:**
- Simple deployment
- Cross-platform compatibility
- Minimal dependencies
- CPU inference primary target

**Files activated:**
- `purego/onnx_runner.go`
- `purego/tokenizer.go`

**Build:**
```bash
# Implicit (default)
go build ./purego/example_simple

# Explicit
go build -tags purego ./purego/example_simple
```

**Requirements:**
- ONNX Runtime library (optional, for real models)
- No other dependencies for SimpleBPETokenizer

### `pytorch`

**Description:** Native PyTorch implementation using LibTorch C++ API

**When to use:**
- Maximum performance needed
- GPU acceleration required
- Complex models
- Access to latest PyTorch features

**Files activated:**
- `pytorch/model_runner.go`
- `pytorch/tokenizer.go`

**Build:**
```bash
go build -tags pytorch ./pytorch/example
```

**Requirements:**
- LibTorch C++ library
- Python with transformers (for tokenizer)
- C++ compiler (g++ or clang)

## Build Commands

### Makefile Integration

Update your Makefile:

```makefile
# Build Pure Go version (default)
build-purego:
	@echo "Building Pure Go version..."
	@go build -tags purego -o bin/example_purego ./purego/example_simple

# Build PyTorch version
build-pytorch:
	@echo "Building PyTorch version..."
	@go build -tags pytorch -o bin/example_pytorch ./pytorch/example

# Build all versions
build-all: build-purego build-pytorch

# Run Pure Go
run-purego: build-purego
	@./bin/example_purego

# Run PyTorch
run-pytorch: build-pytorch
	@export LD_LIBRARY_PATH=./third_party/libtorch/lib:$$LD_LIBRARY_PATH && ./bin/example_pytorch
```

### Direct Go Commands

```bash
# Pure Go with SimpleBPE (works immediately)
go build -o bin/simple ./purego/example_simple
./bin/simple

# Pure Go with ONNX
go build -tags purego -o bin/onnx ./purego/example
./bin/onnx

# PyTorch
go build -tags pytorch -o bin/pytorch ./pytorch/example
./bin/pytorch
```

## Project Structure with Build Tags

```
nano-vllm-go/
├── nanovllm/               # Core library (no build tags)
│   ├── config.go
│   ├── sequence.go
│   ├── scheduler.go
│   └── ...
│
├── purego/                 # Pure Go implementation
│   ├── onnx_runner.go     # //go:build purego (optional)
│   ├── tokenizer.go       # No build tag (always available)
│   └── example_simple/
│
├── pytorch/                # PyTorch implementation
│   ├── model_runner.go    # //go:build pytorch
│   ├── tokenizer.go       # //go:build pytorch
│   └── example/
│       └── main.go        # //go:build pytorch
│
└── example/                # Default example (no build tag)
    └── main.go
```

## Adding Build Tags to Files

### Format

```go
//go:build tagname
// +build tagname

package mypackage
```

**Note:**
- The `//go:build` line is the modern syntax (Go 1.17+)
- The `// +build` line is legacy but still recommended for compatibility
- Both must be at the top of the file
- Blank line required after build tags

### Examples

**Pure Go file:**
```go
//go:build purego
// +build purego

package purego

import "nano-vllm-go/nanovllm"

func NewRunner() nanovllm.ModelRunner {
    // Pure Go implementation
}
```

**PyTorch file:**
```go
//go:build pytorch
// +build pytorch

package pytorch

/*
#cgo LDFLAGS: -ltorch
#include <torch/torch.h>
*/
import "C"

func NewRunner() nanovllm.ModelRunner {
    // PyTorch CGo implementation
}
```

## Conditional Compilation

### Multiple Tags (AND)

```go
//go:build pytorch && linux
// +build pytorch,linux

// Only builds on Linux with pytorch tag
```

### Multiple Tags (OR)

```go
//go:build purego || onnx
// +build purego onnx

// Builds with either purego or onnx tag
```

### Negation

```go
//go:build !pytorch
// +build !pytorch

// Builds when pytorch tag is NOT specified
```

## Testing with Build Tags

```bash
# Test Pure Go implementation
go test -tags purego ./purego/...

# Test PyTorch implementation
go test -tags pytorch ./pytorch/...

# Test all
go test ./...  # Tests files without build tags
```

## Environment Setup

### Pure Go Environment

```bash
# Only need ONNX Runtime for real models
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Build and run
go build -tags purego ./purego/example_simple
./example_simple
```

### PyTorch Environment

```bash
# Need LibTorch
export LD_LIBRARY_PATH=./third_party/libtorch/lib:$LD_LIBRARY_PATH

# Need Python for tokenizer
export PYTHONPATH=/usr/lib/python3.11:$PYTHONPATH

# Build and run
go build -tags pytorch ./pytorch/example
./example
```

## IDE Configuration

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "go.buildTags": "purego",
  "go.testTags": "purego"
}
```

Or for PyTorch:

```json
{
  "go.buildTags": "pytorch",
  "go.testTags": "pytorch"
}
```

### GoLand/IntelliJ

1. Go to **Preferences** → **Go** → **Build Tags & Vendoring**
2. Add `purego` or `pytorch` to **OS/Arch** field

## Common Patterns

### Interface-based Design

```go
// nanovllm/model_runner.go (no build tags)
package nanovllm

type ModelRunner interface {
    Run(seqs []*Sequence, isPrefill bool) ([]int, error)
    Close() error
}
```

```go
// purego/onnx_runner.go
//go:build purego
package purego

// Implements nanovllm.ModelRunner
type ONNXRunner struct { ... }
```

```go
// pytorch/model_runner.go
//go:build pytorch
package pytorch

// Implements nanovllm.ModelRunner
type PyTorchRunner struct { ... }
```

### Factory Pattern

```go
// factory.go (no build tags)
package factory

func NewModelRunner(backend string, config *Config) ModelRunner {
    switch backend {
    case "purego":
        return purego.NewRunner(config)
    case "pytorch":
        return pytorch.NewRunner(config)
    default:
        panic("unknown backend")
    }
}
```

## Debugging

### Check Active Build Tags

```bash
# See what will be compiled
go list -f '{{.GoFiles}}' -tags purego ./purego
go list -f '{{.GoFiles}}' -tags pytorch ./pytorch

# See all tags
go list -f '{{.BuildTags}}' ./...
```

### Verbose Build

```bash
# See what's being compiled
go build -v -tags pytorch ./pytorch/example

# See build constraints
go build -x -tags pytorch ./pytorch/example 2>&1 | grep "build constraint"
```

## Best Practices

1. **Default to Pure Go** - Make Pure Go the default for widest compatibility
2. **Document requirements** - Clearly state what each tag needs
3. **Test both** - CI should test all build tag combinations
4. **Consistent interfaces** - All implementations should satisfy same interface
5. **Separate concerns** - Keep tag-specific code in separate directories
6. **Version constraints** - Document minimum Go version for build tag syntax

## Troubleshooting

### "build constraints exclude all Go files"

**Problem:** No files match the build tag

**Solution:**
```bash
# Check available files
go list -f '{{.GoFiles}}' ./package

# Verify build tag syntax
# Must be at top of file with blank line after
```

### "undefined: SomeFunction"

**Problem:** Function only exists in specific build tag

**Solution:**
```bash
# Make sure to build with correct tag
go build -tags pytorch ./pytorch/example
```

### "multiple packages in directory"

**Problem:** Files with different build tags in same directory

**Solution:**
- Use different package names for different tags
- Or organize into separate directories

## Summary

| Build Tag | Use Case | Dependencies | Performance |
|-----------|----------|--------------|-------------|
| (default) | Quick testing | Minimal | Mock only |
| `purego` | Production (simple) | ONNX Runtime | 70-80% |
| `pytorch` | Production (max perf) | LibTorch | 100% |

**Choose your implementation:**
- **Testing/Demo**: No tags, use SimpleBPETokenizer
- **Production (easy)**: `-tags purego` with ONNX
- **Production (fast)**: `-tags pytorch` with LibTorch

For more details, see:
- `purego/README.md` - Pure Go implementation
- `pytorch/README.md` - PyTorch implementation
- `INTEGRATION.md` - Other integration options
