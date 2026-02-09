# IBM Granite 4 Nano Support

This guide covers support for IBM Granite 4 Nano models, which use a hybrid architecture combining transformer attention layers with Mamba2 state-space model layers.

## Overview

IBM Granite 4 Nano models are lightweight language models with innovative hybrid architectures:

### Granite 4.0-H-350M (Nano Small)
- **Parameters**: 340M active parameters
- **Architecture**: 4 attention layers + 28 Mamba2 layers
- **Context**: 32K tokens
- **Memory**: ~70% less than comparable transformer-only models
- **Speed**: ~2x faster inference

### Granite 4.0-H-1B (Nano Large)
- **Parameters**: 1.5B active parameters
- **Architecture**: 4 attention layers + 36 Mamba2 layers
- **Context**: 128K tokens
- **Memory**: ~70% less than comparable transformer-only models
- **Speed**: ~2x faster inference

## Architecture Details

### Hybrid Design

Granite uses a unique hybrid approach:

```
Layers:
  [0-3]:  Group-Query Attention (GQA) layers
  [4-31]: Mamba2 State-Space Model layers (350M variant)
  [4-39]: Mamba2 State-Space Model layers (1B variant)
```

### Key Features

1. **Group-Query Attention (GQA)**
   - 12 query heads
   - 4 key-value heads
   - Reduces memory while maintaining quality

2. **Mamba2 State-Space Models**
   - Linear O(N) complexity (vs O(N²) for attention)
   - Constant state size (128 dimensions)
   - Content-selective filtering via input-dependent parameters
   - 48 SSM heads with 8 groups

3. **No Positional Encoding (NoPE)**
   - Position-independent embeddings
   - Position information learned implicitly

4. **RMSNorm + SwiGLU**
   - Efficient normalization
   - Advanced activation functions

## Implementation Status

### ✅ Completed

- [x] Mamba2 layer implementation (`purego/tensor/mamba2.go`)
- [x] Hybrid architecture support in config
- [x] Granite configuration presets (350M, 1B)
- [x] Generic model runner with Mamba2 support
- [x] Universal download script with Granite support
- [x] Updated example with Granite models

### 🚧 In Progress

- [ ] Granite-specific weight loader mapping
- [ ] Testing with real Granite models
- [ ] Mamba2 KV cache optimization
- [ ] Chunked processing for long sequences

### 📋 Planned

- [ ] Mamba2 CUDA kernels for acceleration
- [ ] Hybrid attention/Mamba2 parallelization
- [ ] Extended context support (128K tokens)

## Downloading Granite Models

Use the universal download script:

```bash
# Granite 4 Nano 350M (recommended for testing)
./scripts/download_model.py \
  --model ibm-granite/granite-4.0-h-350m \
  --output ./models/granite-350m

# Granite 4 Nano 1B (better quality, requires more memory)
./scripts/download_model.py \
  --model ibm-granite/granite-4.0-h-1b \
  --output ./models/granite-1b \
  --fp16
```

### Model Variants

Each Granite size has two variants:

- **Base models**: `granite-4.0-h-{size}-base`
  - Pre-trained foundation models
  - Good for fine-tuning

- **Instruct models**: `granite-4.0-h-{size}`
  - Instruction-tuned for chat/tasks
  - Ready to use out of the box

## Running Granite Models

Once downloaded, run with the generic example:

```bash
# Auto-detect and run Granite model
MODEL_DIR=./models/granite-350m ./bin/generic_test

# Or with custom prompts
MODEL_DIR=./models/granite-350m ./bin/generic_test "Explain quantum computing"

# Multiple prompts
MODEL_DIR=./models/granite-350m ./bin/generic_test \
  "What is AI?" \
  "Write a hello world in Python" \
  "Explain transformers"
```

## Mamba2 Architecture

### Core Concept

Mamba2 is a selective state-space model that replaces quadratic attention with linear recurrence:

**Traditional Attention:**
```
Complexity: O(N²)
Memory: O(N²) for attention matrix
Operation: Compare all token pairs
```

**Mamba2 SSM:**
```
Complexity: O(N)
Memory: O(N × state_size) constant state
Operation: Sequential state updates with content filtering
```

### Mathematical Foundation

The state-space model at each timestep:

```
State update:  x[n+1] = A_bar × x[n] + B[n] × u[n]
Output:        y[n] = C[n] × x[n] + D × u[n]

Where:
  x[n]   - Hidden state (128 dimensions)
  u[n]   - Input at timestep n
  y[n]   - Output at timestep n
  A_bar  - Discretized state transition (depends on Δ[n])
  B[n]   - Input-dependent projection
  C[n]   - Input-dependent projection
  D      - Skip connection
  Δ[n]   - Input-dependent time step
```

**Key Innovation**: B, C, and Δ are computed from the input, enabling content-selective filtering.

### Mamba2 Components

1. **State Size**: 128 (constant memory per head)
2. **Number of Heads**: 48 (parallel SSM computations)
3. **Number of Groups**: 8 (B,C sharing across heads)
4. **Causal Convolution**: 4-token local context
5. **Chunk Size**: 256 tokens (for efficient processing)

## Performance Characteristics

### Memory Usage

Granite Nano models use significantly less memory than transformer-only models:

| Model | Parameters | Peak Memory | Context Length |
|-------|-----------|-------------|----------------|
| GPT-2 (124M) | 124M | ~500MB | 1K |
| **Granite 350M** | **340M** | **~400MB** | **32K** |
| Llama 2 (7B) | 7B | ~14GB | 4K |
| **Granite 1B** | **1.5B** | **~2GB** | **128K** |

### Inference Speed

Mamba2 layers are faster than attention, especially for long sequences:

- **Short sequences (< 1K)**: ~10% faster
- **Medium sequences (1K-32K)**: ~2x faster
- **Long sequences (> 32K)**: ~5-10x faster

### Use Cases

Granite models excel at:

- **Code completion** with long context
- **Document understanding** (long files)
- **Chat with extended history**
- **Low-memory deployment** (edge devices)
- **Fast inference** at any sequence length

## Configuration Reference

### Granite 350M Config

```go
config := tensor.NewGraniteConfig("350m")
// Hidden: 768
// Layers: 32 (4 attention + 28 Mamba2)
// Attention heads: 12 (query), 4 (KV)
// Mamba2 heads: 48
// State size: 128
// Context: 32K tokens
```

### Granite 1B Config

```go
config := tensor.NewGraniteConfig("1b")
// Hidden: 1536
// Layers: 40 (4 attention + 36 Mamba2)
// Attention heads: 12 (query), 4 (KV)
// Mamba2 heads: 48
// State size: 128
// Context: 128K tokens
```

## Advanced Usage

### Programmatic Access

```go
import (
    "nano-vllm-go/nanovllm"
    "nano-vllm-go/purego"
    "nano-vllm-go/purego/tensor"
)

// Create Granite-specific config
config := nanovllm.NewConfig(
    ".",
    nanovllm.WithMaxNumSeqs(16),
    nanovllm.WithMaxNumBatchedTokens(2048),
)

// Load Granite model (auto-detects architecture)
modelRunner, err := purego.NewGenericModelRunner(
    "./models/granite-350m",
    config,
)

// Or create with explicit Granite config
graniteConfig := tensor.NewGraniteConfig("350m")
modelRunner, err := purego.NewGenericModelRunnerFromConfig(
    "./models/granite-350m/model.safetensors",
    graniteConfig,
    config,
)

// Load tokenizer
tokenizer, err := purego.NewUniversalTokenizer("./models/granite-350m")

// Create LLM engine
llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)

// Generate
outputs, err := llm.GenerateSimple(
    []string{"Explain Mamba2 architecture"},
    nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(200),
    ),
    true,
)
```

### Model Info

After loading, you can inspect the model configuration:

```go
modelConfig := modelRunner.GetModelConfig()

fmt.Printf("Model: %s\n", modelConfig.ModelName)
fmt.Printf("Architecture: %s\n", modelConfig.Architecture)
fmt.Printf("Layers: %d total\n", modelConfig.NumLayers)
fmt.Printf("  - Attention: %d\n", modelConfig.NumAttentionLayers)
fmt.Printf("  - Mamba2: %d\n", modelConfig.NumMamba2Layers)
fmt.Printf("Attention: %s (%d query, %d KV heads)\n",
    modelConfig.AttentionType,
    modelConfig.NumHeads,
    modelConfig.NumKVHeads)
fmt.Printf("Mamba2: %d heads, %d state size\n",
    modelConfig.Mamba2NumHeads,
    modelConfig.Mamba2StateSize)
```

## Troubleshooting

### Model Download Fails

```bash
# Ensure you have the required packages
pip install transformers safetensors torch

# For Granite models, you may need to accept license on HuggingFace
# Visit: https://huggingface.co/ibm-granite/granite-4.0-h-350m
```

### Out of Memory

```bash
# Use float16 for 1B model
./scripts/download_model.py \
  --model ibm-granite/granite-4.0-h-1b \
  --output ./models/granite-1b \
  --fp16

# Or use smaller 350M model
./scripts/download_model.py \
  --model ibm-granite/granite-4.0-h-350m \
  --output ./models/granite-350m
```

### Weights Not Loading

The Granite weight loader is still being tested. If you encounter weight loading issues:

1. Check that `config.json` is present in model directory
2. Verify `model.safetensors` exists
3. Check the tensor names in the safetensors file match expected patterns

## References

### Papers

1. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
   - Authors: Albert Gu, Tri Dao
   - arXiv: 2312.00752 (December 2023)
   - Introduces selective SSMs

2. **Transformers are SSMs: Generalized Models and Efficient Algorithms**
   - Authors: Tri Dao, Albert Gu
   - arXiv: 2405.21060 (May 2024)
   - Mamba2 architecture and SSD theory

### Resources

- **Granite Models**: https://huggingface.co/ibm-granite
- **Mamba Repository**: https://github.com/state-spaces/mamba
- **HF Transformers Mamba2**: https://huggingface.co/docs/transformers/model_doc/mamba2

## Contributing

Granite/Mamba2 support is actively being developed. Contributions welcome:

- Weight loader improvements
- Performance optimizations
- Extended context testing
- Comparison benchmarks

See `ARCHITECTURE_COMPARISON.md` for technical details on different architectures.
