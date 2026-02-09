# Running Falcon 7B - Implementation Guide

## What We Added

To support Falcon 7B, we added **3 key components** (~300 lines of code):

### 1. RoPE (Rotary Position Embeddings)
**File:** `purego/tensor/rope.go` (~130 lines)

```go
// Instead of learned position embeddings (GPT-2):
x = TokenEmbedding[token] + PosEmbedding[position]

// RoPE rotates Q and K based on position:
ropeCache := NewRoPECache(headDim, maxSeqLen, 10000.0)
ropeCache.ApplyRoPE(Q, K, position)
```

**Why better:**
- No learned parameters
- Works for any sequence length
- Better for long context

### 2. Multi-Query Attention (MQA)
**File:** `purego/tensor/mqa.go` (~180 lines)

```go
// GPT-2: Each head gets separate K,V
// 12 heads = 12 × (Q + K + V) = 36 weight matrices

// Falcon: All heads share K,V
// 71 heads = 71 × Q + 1 × (K + V) = 73 weight matrices (vs 213!)
```

**Memory savings:**
- GPT-2: KV cache = 2 × num_heads × seq_len × head_dim
- Falcon: KV cache = 2 × 1 × seq_len × head_dim
- **71x less KV cache memory!**

### 3. Parallel Attention + FFN
**File:** `purego/tensor/falcon.go` (~120 lines)

```go
// GPT-2: Sequential
residual = x
x = LayerNorm(x)
x = Attention(x)
x = x + residual

residual = x
x = LayerNorm(x)
x = FFN(x)
x = x + residual

// Falcon: Parallel
residual = x
x = LayerNorm(x)
attn_out = Attention(x)
ffn_out = FFN(x)
x = residual + attn_out + ffn_out  // Both at once!
```

**Why faster:**
- Can parallelize attention + FFN
- Fewer sequential dependencies
- Better GPU utilization

## Falcon 7B Architecture

```
Falcon 7B Specs:
- Parameters: 7.0 billion
- Layers: 32
- Hidden size: 4544
- Attention heads: 71 (Multi-Query!)
- Head dimension: 64
- FFN dimension: 18176
- Vocabulary: 65024
- Max sequence: 2048

Memory Requirements:
- Weights: ~14 GB (FP32) or ~7 GB (FP16)
- KV cache (per token): ~1.2 MB
- Activations: ~200 MB
- Total: ~8-15 GB
```

## What Still Needs to Be Done

### 1. Weight Loader (30 minutes)

Add Falcon-specific weight loading:

```go
// purego/tensor/falcon_loader.go
func LoadFalconFromSafetensors(path string, config *FalconConfig) (*FalconModel, error) {
    model := NewFalconModel(config)

    // Load token embedding
    loadTensor(data, metadata, "transformer.word_embeddings.weight", &model.TokenEmbedding)

    // Load each layer
    for i := 0; i < config.NumLayers; i++ {
        prefix := fmt.Sprintf("transformer.h.%d", i)
        block := model.Blocks[i]

        // MQA weights
        loadTensor(data, metadata, prefix+".self_attention.query_key_value.weight", ...)
        // Split into Q (large) and KV (small)

        // FFN weights
        loadTensor(data, metadata, prefix+".mlp.dense_h_to_4h.weight", &block.FFN.W1)
        loadTensor(data, metadata, prefix+".mlp.dense_4h_to_h.weight", &block.FFN.W2)

        // Layer norms
        loadTensor(data, metadata, prefix+".input_layernorm.weight", &block.InputLN.Weight)
        loadTensor(data, metadata, prefix+".input_layernorm.bias", &block.InputLN.Bias)
    }

    // Final layer norm and LM head
    loadTensor(data, metadata, "transformer.ln_f.weight", &model.LNFinal.Weight)
    loadTensor(data, metadata, "lm_head.weight", &model.LMHead)

    return model, nil
}
```

### 2. ModelRunner Integration (15 minutes)

```go
// purego/falcon_runner.go
type FalconModelRunner struct {
    model       *tensor.FalconModel
    initialized bool
}

func NewFalconModelRunner(modelPath string, config *nanovllm.Config) (*FalconModelRunner, error) {
    falconConfig := &tensor.FalconConfig{
        VocabSize:  65024,
        Hidden:     4544,
        NumLayers:  32,
        NumHeads:   71,
        FFNDim:     18176,
        MaxSeqLen:  2048,
        EOSTokenID: 11,
    }

    model, err := tensor.LoadFalconFromSafetensors(modelPath, falconConfig)
    if err != nil {
        return nil, err
    }

    return &FalconModelRunner{
        model:       model,
        initialized: true,
    }, nil
}

func (m *FalconModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    tokenIDs := make([]int, len(seqs))

    for i, seq := range seqs {
        logits := m.model.Forward(seq.TokenIDs)
        lastTokenLogits := m.model.GetLogitsForLastToken(logits)
        tokenIDs[i] = sampleToken(lastTokenLogits, seq.Temperature)
    }

    return tokenIDs, nil
}
```

### 3. Download Script (10 minutes)

```python
# scripts/download_falcon.py
#!/usr/bin/env python3
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b")

# Save in safetensors format
state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, "./models/falcon-7b/model.safetensors")

tokenizer.save_pretrained("./models/falcon-7b")
```

### 4. Critical: KV Cache (~100 lines)

**Without KV cache, Falcon 7B will be unusably slow!**

```go
// Each decode step recomputes ALL previous tokens
// Token 1: Process 1 token
// Token 2: Process 1+2 = 3 tokens
// Token 3: Process 1+2+3 = 6 tokens
// Token 100: Process 1+2+...+100 = 5050 tokens!
// This is O(n²) complexity!

// With KV cache:
type KVCache struct {
    Keys   *Tensor  // [num_layers, batch, 1, seq_len, head_dim]
    Values *Tensor  // [num_layers, batch, 1, seq_len, head_dim]
}

// Each decode step only processes 1 new token
// Token 1: Process 1 token, cache K,V
// Token 2: Process 1 token, use cached K,V from token 1
// Token 3: Process 1 token, use cached K,V from tokens 1,2
// This is O(n) complexity!
```

**Implementation sketch:**

```go
func (mqa *MultiQueryAttention) ForwardWithCache(
    x *Tensor,
    cache *KVCache,
    position int,
) *Tensor {
    // Only compute Q for new token
    Q := mqa.projectQ(x)  // [batch, 1, hidden]

    // Only compute K,V for new token
    K_new, V_new := mqa.projectKV(x)  // [batch, 1, head_dim]

    // Append to cache
    cache.Keys = append(cache.Keys, K_new)
    cache.Values = append(cache.Values, V_new)

    // Use ALL cached K,V for attention
    K_all := cache.Keys   // [batch, 1, position+1, head_dim]
    V_all := cache.Values

    // Attention only on new token but attending to all past
    output := scaledDotProduct(Q, K_all, V_all)

    return output
}
```

## Performance Estimates

### Without KV Cache (Current)
```
Falcon 7B (32 layers, 4544 hidden)
CPU: Pure Go naive matmul

Token 1:   ~0.5 seconds
Token 10:  ~5 seconds (10x slower!)
Token 50:  ~25 seconds (50x slower!)
Token 100: ~50 seconds (100x slower!)

Total for 100 tokens: ~2500 seconds = 42 minutes!
Throughput: ~0.04 tok/s
```

### With KV Cache
```
Token 1:   ~0.5 seconds
Token 10:  ~0.5 seconds (constant!)
Token 50:  ~0.5 seconds
Token 100: ~0.5 seconds

Total for 100 tokens: ~50 seconds
Throughput: ~2 tok/s
```

**50x speedup from KV cache!**

### With Optimizations

```
Layer          | Speedup | Cumulative
---------------|---------|------------
KV cache       | 50x     | 50x
gonum/BLAS     | 10x     | 500x
Quantization   | 2x      | 1000x
Flash Attention| 2x      | 2000x
---------------|---------|------------
Final: ~40 tok/s (vs 0.04 tok/s baseline)
```

## Memory Requirements

### Falcon 7B

**Model Weights:**
- FP32: 7B × 4 bytes = ~28 GB ❌ Too big!
- FP16: 7B × 2 bytes = ~14 GB ⚠️ Barely fits
- INT8: 7B × 1 byte = ~7 GB ✅ Reasonable
- INT4: 7B × 0.5 bytes = ~3.5 GB ✅ Great!

**KV Cache (with MQA!):**
- Per token: 2 × 32 layers × 1 head × 64 dim × 4 bytes = 16 KB
- For 2048 context: 16 KB × 2048 = 33 MB ✅ Totally fine!
- (vs Multi-Head: 71 heads = 2.3 GB ❌)

**This is why Falcon uses MQA!**

### Comparison

```
Model      | Params | MHA KV Cache | MQA KV Cache | Speedup
-----------|--------|--------------|--------------|--------
GPT-2      | 124M   | 50 MB        | 4 MB         | 12.5x
Falcon 7B  | 7.0B   | 2.3 GB       | 33 MB        | 71x
Falcon 40B | 40B    | 13 GB        | 183 MB       | 71x
```

## Complete Implementation Checklist

- [x] RoPE implementation
- [x] Multi-Query Attention
- [x] Parallel attention + FFN architecture
- [x] Falcon model structure
- [ ] Falcon weight loader (~30 min)
- [ ] ModelRunner integration (~15 min)
- [ ] Download script (~10 min)
- [ ] **KV cache implementation (~2 hours) - CRITICAL!**
- [ ] Quantization (INT8/INT4) (~1 day)
- [ ] BLAS integration (~2 days)
- [ ] Flash Attention (~3 days)

## Realistic Timeline

### Minimum Viable (Can run, but slow)
**Time:** 1-2 hours
- ✅ Architecture (done!)
- ⏱️ Weight loader (30 min)
- ⏱️ ModelRunner (15 min)
- ⏱️ Download script (10 min)

**Result:** Can load and run Falcon 7B at ~0.04 tok/s

### Usable (Reasonable speed)
**Time:** +4 hours
- ⏱️ KV cache (2 hours) - **ESSENTIAL**
- ⏱️ Simple optimizations (2 hours)

**Result:** ~2 tok/s (50x speedup)

### Fast (Production quality)
**Time:** +2 weeks
- Quantization (INT8)
- gonum/BLAS integration
- Memory optimizations

**Result:** ~20-40 tok/s

## Try It Now

Want to implement this? Here's the order:

1. **Weight Loader** (30 min) - So we can actually load the model
2. **Download Script** (10 min) - Get Falcon weights
3. **ModelRunner** (15 min) - Integrate with nano-vllm
4. **Test** (5 min) - Run inference (will be slow!)
5. **KV Cache** (2 hours) - Make it usable
6. **Optimize** (ongoing) - Make it fast

## Summary

**What we have:**
- ✅ Complete Falcon architecture
- ✅ RoPE position embeddings
- ✅ Multi-Query Attention (71 heads, 1 KV)
- ✅ Parallel attention + FFN
- ✅ Ready to load weights

**What we need:**
- Weight loading (30 min)
- ModelRunner integration (15 min)
- KV cache (2 hours) - **Critical for usability**
- Optimizations (days-weeks) - For speed

**Complexity:**
- Architecture: ✅ Done (~300 lines)
- Making it work: ~1 hour
- Making it usable: ~4 hours
- Making it fast: ~2 weeks

**Bottom line:** You're ~1 hour away from running Falcon 7B, ~4 hours from it being usable!

Want me to implement the weight loader and finish it? 🚀
