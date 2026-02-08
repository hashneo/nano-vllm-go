# Nano-vLLM-Go Architecture

This document explains the architecture of nano-vllm-go and how the different components work together.

## Overview

Nano-vLLM-Go is a Go implementation of the core scheduling and memory management logic from vLLM. The architecture is designed to maximize GPU utilization through continuous batching and prefix caching.

## Key Components

### 1. Sequence (`sequence.go`)

Represents a single generation request with its associated state:

- **Token Management**: Tracks prompt tokens vs completion tokens
- **Block Allocation**: Maintains a block table for KV cache
- **Status Tracking**: WAITING → RUNNING → FINISHED
- **Metadata**: Temperature, max tokens, and other sampling parameters

```go
type Sequence struct {
    SeqID             int64
    Status            SequenceStatus
    TokenIDs          []int
    NumTokens         int
    NumPromptTokens   int
    NumCachedTokens   int
    BlockTable        []int
    // ... sampling parameters
}
```

### 2. Block Manager (`block_manager.go`)

Manages KV cache memory with prefix caching:

- **Block-based Allocation**: Memory is allocated in fixed-size blocks (default: 256 tokens)
- **Reference Counting**: Blocks can be shared between sequences
- **Content-based Hashing**: Uses xxhash to identify identical prefixes
- **Lazy Deallocation**: Blocks stay in cache even after deallocation for reuse

**Key Insight**: When two sequences share a common prefix (e.g., system prompts), they share the same KV cache blocks, dramatically reducing memory usage.

```go
// Example: Two sequences with same 256-token prefix
seq1: [common_prefix_256] [unique_tokens_100]
seq2: [common_prefix_256] [unique_tokens_200]

// Both sequences reference the same block for common_prefix_256
```

### 3. Scheduler (`scheduler.go`)

Coordinates batch execution with two phases:

#### Prefill Phase
- Processes input tokens for new requests
- Can batch multiple sequences together
- Token count: sum of all input tokens
- Computationally intensive (proportional to sequence length²)

#### Decode Phase
- Generates one token per sequence
- All running sequences generate simultaneously
- Token count: number of sequences (1 token each)
- Bounded by max batch size

**Scheduling Algorithm**:
1. Try to schedule waiting sequences for prefill (if space available)
2. If no prefill work, schedule running sequences for decode
3. Preempt sequences if memory is full (deallocate KV cache)
4. Preempted sequences return to waiting queue

### 4. Model Runner (`model_runner.go`)

Interface for model inference - the actual "AI brain":

```go
type ModelRunner interface {
    Run(seqs []*Sequence, isPrefill bool) ([]int, error)
    Close() error
}
```

The mock implementation is provided for demonstration. In production, implement this using:

- **PyTorch via CGo**: Call into Python/C++ libraries
- **ONNX Runtime**: Use Go bindings
- **HTTP/gRPC**: Call remote inference servers
- **TensorRT**: Use NVIDIA's inference runtime
- **Custom CUDA**: Write custom kernels

### 5. LLM Engine (`llm_engine.go`)

Orchestrates the entire inference process:

```go
Generate flow:
1. AddRequest() - tokenize and queue prompts
2. Loop until finished:
   a. Schedule() - select sequences for this step
   b. ModelRunner.Run() - execute inference
   c. Postprocess() - append tokens, check completion
3. Return completed outputs
```

### 6. Tokenizer Interface (`model_runner.go`)

Abstraction for text ↔ tokens conversion:

```go
type Tokenizer interface {
    Encode(text string) ([]int, error)
    Decode(tokenIDs []int) (string, error)
    EOSTokenID() int
}
```

Implement using:
- **Hugging Face Tokenizers**: Via CGo or HTTP API
- **SentencePiece**: Go bindings available
- **BPE**: Pure Go implementations exist
- **TikToken**: For OpenAI models

## Memory Management

### Block Allocation Example

```
Sequence with 600 tokens, block size = 256:

Block 0: [tok_0   ... tok_255  ] ← Full block, hash = H1
Block 1: [tok_256 ... tok_511  ] ← Full block, hash = H2
Block 2: [tok_512 ... tok_599  ] ← Partial block, hash = None

Block table: [block_id_0, block_id_1, block_id_2]
```

### Prefix Caching Example

```
System prompt: "You are a helpful assistant." (256 tokens)
User queries: "What is AI?", "Explain quantum computing."

Both queries share the same system prompt block:
- Block manager computes hash of first 256 tokens
- Second query finds cached block with matching hash
- Reuses KV cache block (increments ref count)
- Only processes unique tokens

Memory saved: 256 tokens × 2 layers × 128 dims × 16 bytes = ~1MB per sequence
```

## Scheduling Strategy

### Continuous Batching

Traditional batching waits for all sequences to complete:
```
Batch 1: [seq_A, seq_B, seq_C] → wait for all to finish
Batch 2: [seq_D, seq_E, seq_F] → wait for all to finish
```

Continuous batching removes sequences as they complete:
```
Step 1: [A, B, C]
Step 2: [A, B, C]
Step 3: [A, C, D]     ← B finished, D added
Step 4: [A, C, D, E]  ← E added
Step 5: [C, D, E]     ← A finished
```

**Benefits**:
- Higher GPU utilization
- Lower latency for short sequences
- Better throughput overall

### Preemption

When memory is full, scheduler can preempt sequences:
1. Deallocate KV cache blocks
2. Move sequence back to waiting queue
3. Blocks stay in cache for potential reuse
4. Sequence will be rescheduled later

## Performance Considerations

### Batch Size vs Latency

- **Small batches**: Lower latency, lower throughput
- **Large batches**: Higher throughput, higher latency
- Configure via `MaxNumSeqs` and `MaxNumBatchedTokens`

### Prefill vs Decode

- **Prefill**: O(n²) complexity, memory-intensive
- **Decode**: O(1) per token, compute-bound
- Separating phases allows better scheduling

### Memory Tuning

- `KVCacheBlockSize`: Larger = less overhead, less granular sharing
- `NumKVCacheBlocks`: More = can serve more concurrent requests
- `GPUMemoryUtilization`: How much GPU memory to use for KV cache

## Integration Guide

### 1. Implement ModelRunner

```go
type MyModelRunner struct {
    model *YourModelType
}

func (m *MyModelRunner) Run(seqs []*Sequence, isPrefill bool) ([]int, error) {
    // Prepare input tensors
    // Run model forward pass
    // Sample next tokens
    // Return token IDs
}
```

### 2. Implement Tokenizer

```go
type MyTokenizer struct {
    tokenizer *YourTokenizerType
}

func (t *MyTokenizer) Encode(text string) ([]int, error) {
    // Tokenize text
    // Return token IDs
}
```

### 3. Create LLM

```go
config := nanovllm.NewConfig("/path/to/model")
modelRunner := NewMyModelRunner(config)
tokenizer := NewMyTokenizer(config)

llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
defer llm.Close()
```

## Future Extensions

### Tensor Parallelism

Split model across multiple GPUs:
```go
// Coordinator goroutine
for rank := 0; rank < config.TensorParallelSize; rank++ {
    go func(rank int) {
        // Each GPU runs same computation on split tensors
        // All-reduce to combine results
    }(rank)
}
```

### Speculative Decoding

Generate multiple tokens in parallel then verify:
```
Draft model: Generate N tokens quickly
Target model: Verify in parallel
Accept: Take longest correct prefix
```

### Quantization Support

Add support for INT8/INT4 quantized models to reduce memory.

### Flash Attention Integration

Use Flash Attention for faster attention computation.

## Comparison with Python Version

| Feature | Python nano-vllm | Go nano-vllm-go |
|---------|-----------------|-----------------|
| Core logic | ✅ Same | ✅ Same |
| Model inference | PyTorch | Interface (implement) |
| Tokenization | HF Transformers | Interface (implement) |
| Tensor parallelism | multiprocessing | Goroutines + interface |
| Dependencies | torch, triton | xxhash, progressbar |
| Lines of code | ~1200 | ~1000 |

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
go test ./...

# Run with coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out

# Run specific test
go test -v -run TestBlockManager ./nanovllm
```

## Benchmarking

```bash
# Build benchmark
go build -o bin/bench ./bench

# Run benchmark
./bin/bench
```

Note: The benchmark uses mock inference, so performance numbers represent scheduling overhead only, not actual model inference time.

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [Original nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)
- [Paged Attention](https://arxiv.org/abs/2309.06180)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
