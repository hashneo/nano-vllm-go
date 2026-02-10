# Available Model Architectures

This project includes implementations of multiple transformer architectures for educational purposes.

## Supported Architectures

### 1. **GPT-2** ✅ (Fully Implemented & Tested)
- **File**: `purego/tensor/gpt2.go`, `purego/tensor/loader.go`
- **Attention**: Multi-Head Attention (MHA)
- **Position**: Learned embeddings
- **Activation**: GELU
- **Normalization**: LayerNorm
- **Block Style**: Sequential (Attn → FFN)
- **Context**: 1024 tokens
- **Sizes**: 124M (Small), 355M (Medium), 774M (Large), 1.5B (XL)

**Key Features**:
- 12 attention heads (Small)
- 768 hidden dimensions
- Byte-level BPE tokenization
- Works with actual HuggingFace models

**Try it**:
```bash
./ask-gpt2 "The capital city of France is"
```

---

### 2. **Falcon** (Implementation Available)
- **Files**: `purego/tensor/falcon.go`, `purego/tensor/mqa.go`, `purego/tensor/rope.go`
- **Attention**: Multi-Query Attention (MQA)
- **Position**: RoPE (Rotary Position Embedding)
- **Activation**: GELU
- **Normalization**: LayerNorm
- **Block Style**: Parallel (Attn ‖ FFN)
- **Context**: 2048 tokens
- **Sizes**: 7B, 40B parameters

**Key Innovations**:
- **MQA**: Single key-value head shared across all query heads
  - Reduces KV cache memory by ~8-12×
  - Faster inference at slight quality cost
- **RoPE**: Rotary position encoding (no learned embeddings)
- **Parallel Blocks**: Attention and FFN computed in parallel

**Learn About**:
- Multi-Query Attention (`purego/tensor/mqa.go`)
- RoPE implementation (`purego/tensor/rope.go`)
- Parallel transformer blocks (`purego/tensor/falcon.go`)

---

### 3. **Llama** (Implementation Available)
- **Files**: Configuration in `purego/tensor/config.go`
- **Attention**: Grouped-Query Attention (GQA)
- **Position**: RoPE
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Block Style**: Sequential
- **Context**: 4096 tokens
- **Sizes**: 7B, 13B parameters

**Key Innovations**:
- **GQA**: Compromise between MHA and MQA
  - 8 KV heads for 32 query heads (4:1 ratio)
  - Better quality than MQA, more efficient than MHA
- **SwiGLU**: Gated activation function (better than GELU)
- **RMSNorm**: Simpler normalization (no mean subtraction)

**Comparison**:
```
MHA (GPT-2):  32 Q heads, 32 KV heads → Most memory
GQA (Llama):  32 Q heads,  8 KV heads → Balanced
MQA (Falcon): 32 Q heads,  1 KV head  → Least memory
```

---

### 4. **Granite** (IBM Hybrid Architecture)
- **Files**: Configuration and Mamba2 implementation
- **Type**: Hybrid (Attention + Mamba2)
- **Attention**: GQA (sparse, only 4 layers)
- **Position**: None (Mamba2 doesn't need it)
- **Activation**: SwiGLU
- **Normalization**: RMSNorm
- **Context**: 32K-128K tokens
- **Sizes**: 350M, 1B parameters

**Key Innovation**:
- **Hybrid Architecture**: 4 attention layers + 28 Mamba2 layers
- **Mamba2**: State Space Model (alternative to attention)
  - O(N) complexity instead of O(N²)
  - Better for long sequences
- **No positional encoding needed**

**Layer Distribution** (350M):
```
Layers 0-9:   Mamba2
Layer 10:     Attention
Layers 11-12: Mamba2
Layer 13:     Attention
Layers 14-16: Mamba2
Layer 17:     Attention
Layers 18-26: Mamba2
Layer 27:     Attention
Layers 28-31: Mamba2
```

---

## Architecture Comparison

| Feature | GPT-2 | Falcon | Llama | Granite |
|---------|-------|--------|-------|---------|
| **Attention** | MHA | MQA | GQA | GQA (sparse) |
| **Position** | Learned | RoPE | RoPE | None |
| **Norm** | LayerNorm | LayerNorm | RMSNorm | RMSNorm |
| **Activation** | GELU | GELU | SwiGLU | SwiGLU |
| **Block Style** | Sequential | Parallel | Sequential | Hybrid |
| **KV Cache** | Large | Small | Medium | Small |
| **Complexity** | O(N²) | O(N²) | O(N²) | O(N) |
| **Context** | 1K | 2K | 4K | 32K-128K |

---

## Key Technologies Explained

### Multi-Head Attention (MHA)
```
Q: [batch, heads, seq, head_dim]  # 12 heads
K: [batch, heads, seq, head_dim]  # 12 heads
V: [batch, heads, seq, head_dim]  # 12 heads
```
Each head has its own Q, K, V. Most memory, best quality.

### Multi-Query Attention (MQA)
```
Q: [batch, heads, seq, head_dim]  # 12 heads
K: [batch, 1, seq, head_dim]      # 1 head (shared)
V: [batch, 1, seq, head_dim]      # 1 head (shared)
```
All query heads share one K, V. Least memory, faster inference.

### Grouped-Query Attention (GQA)
```
Q: [batch, heads, seq, head_dim]  # 32 heads
K: [batch, groups, seq, head_dim] # 8 groups
V: [batch, groups, seq, head_dim] # 8 groups
```
Query heads grouped, each group shares K, V. Best balance.

### RoPE (Rotary Position Embedding)
- Encodes position by rotating query/key vectors
- No learned embeddings needed
- Better extrapolation to longer sequences
- Used in Falcon, Llama, most modern LLMs

### Block Styles

**Sequential** (GPT-2, Llama):
```
x → LayerNorm → Attention → + → LayerNorm → FFN → + → out
    ↑__________________________|  ↑___________________|
```

**Parallel** (Falcon):
```
x → LayerNorm → [Attention]
                     ↓
                 [   +   ] → out
                     ↑
x ─────────────→ [  FFN  ]
```

---

## Code Structure

### For GPT-2 (Working Implementation)
```
ask_gpt2.go              # Main program
↓
nanovllm/llm.go          # User API
↓
nanovllm/llm_engine.go   # Orchestration
↓
nanovllm/tensor_model_runner.go  # GPT-2 specific
↓
purego/tensor/gpt2.go    # GPT-2 transformer
purego/tensor/loader.go  # GPT-2 weight loading
```

### For Generic Architectures (Educational)
```
purego/tensor/generic_model.go   # Universal transformer
purego/tensor/generic_loader.go  # Universal weight loading
↓
purego/tensor/config.go          # Architecture configs
├─ NewGPT2Config()
├─ NewFalconConfig()
├─ NewLlamaConfig()
└─ NewGraniteConfig()
```

---

## Learning Path

1. **Start with GPT-2** (`purego/tensor/gpt2.go`)
   - Understand basic transformer architecture
   - Multi-head attention
   - Feed-forward networks
   - Layer normalization

2. **Study MQA** (`purego/tensor/mqa.go`)
   - See how memory is reduced
   - Compare with standard attention
   - Understand trade-offs

3. **Explore RoPE** (`purego/tensor/rope.go`)
   - Alternative to learned positions
   - Rotation-based encoding
   - Better extrapolation

4. **Compare Architectures** (`purego/tensor/config.go`)
   - See all configurations side-by-side
   - Understand design choices
   - Learn evolution of LLM architectures

5. **Read ARCHITECTURE_GUIDE.md**
   - Comprehensive guide with diagrams
   - GPT evolution (GPT-2 to GPT-5)
   - Detailed explanations of each component

---

## Files to Study

### Core Implementations
- `purego/tensor/gpt2.go` - GPT-2 model
- `purego/tensor/attention.go` - Multi-head attention
- `purego/tensor/mqa.go` - Multi-query attention (Falcon)
- `purego/tensor/rope.go` - Rotary position encoding
- `purego/tensor/falcon.go` - Falcon model
- `purego/tensor/mamba2.go` - Mamba2 state space model (Granite)

### Configuration
- `purego/tensor/config.go` - All architecture configs
- `purego/tensor/generic_model.go` - Universal transformer
- `purego/tensor/generic_loader.go` - Universal weight loading

### Documentation
- `ARCHITECTURE_GUIDE.md` - Comprehensive guide
- `README.md` - Quick start guide

---

## References

- **GPT-2**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **MQA**: [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150)
- **RoPE**: [RoFormer](https://arxiv.org/abs/2104.09864)
- **GQA**: [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245)
- **Llama**: [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- **Falcon**: [Falcon LLM](https://huggingface.co/tiiuae/falcon-7b)
- **Mamba**: [Mamba Paper](https://arxiv.org/abs/2312.00752)

---

## Current Status

✅ **GPT-2**: Fully working with real models
📚 **Falcon**: Code available for study (MQA, RoPE, parallel blocks)
📚 **Llama**: Config available for study (GQA, SwiGLU, RMSNorm)
📚 **Granite**: Hybrid architecture example (Attention + Mamba2)

All architectures are available in the codebase for educational purposes!
