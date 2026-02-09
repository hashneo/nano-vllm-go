# Granite Implementation Status

## Summary

IBM Granite 4 Nano support has been **partially implemented**. The architecture, Mamba2 layers, and download tooling are complete, but the weight loader needs Granite-specific mappings.

## ✅ Completed Components

### 1. Mamba2 State-Space Model Layer
- **File**: `purego/tensor/mamba2.go` (430 lines)
- **Status**: ✅ Complete
- **Features**:
  - Selective state-space model with input-dependent parameters
  - Linear O(N) complexity
  - Causal convolution (4-token kernel)
  - State caching for inference
  - SiLU and Softplus activations

### 2. Granite Architecture Configuration
- **File**: `purego/tensor/config.go`
- **Status**: ✅ Complete
- **Features**:
  - Hybrid layer configuration (attention + Mamba2)
  - 350M variant: 4 attention + 28 Mamba2 layers
  - 1B variant: 4 attention + 36 Mamba2 layers
  - Correct layer pattern (attention at layers 10, 13, 17, 27)
  - Group-Query Attention config (12 query, 4 KV heads)
  - Mamba2 parameters (48 heads, 128 state size, 8 groups)

### 3. Hybrid Model Support
- **File**: `purego/tensor/generic_model.go`
- **Status**: ✅ Complete
- **Features**:
  - `NewGenericBlockWithType()` for attention or Mamba2 layers
  - Auto-detect layer type from `HybridLayers` config
  - Mixed attention/Mamba2 forward passes
  - Mamba2-specific forward without FFN

### 4. Universal Download Script
- **File**: `scripts/download_model.py`
- **Status**: ✅ Complete
- **Features**:
  - Downloads any HuggingFace model
  - Auto-converts to safetensors format
  - Handles tied embeddings (weight sharing)
  - Works with Granite, GPT-2, Falcon, Llama, Mistral, etc.

### 5. Model Downloaded
- **Model**: IBM Granite 4.0-H-350M
- **Size**: 649MB (340M parameters)
- **Location**: `./models/granite-350m/`
- **Files**:
  ```
  ✓ config.json (model config)
  ✓ tokenizer.json (universal tokenizer)
  ✓ model.safetensors (weights)
  ✓ model_info.json (metadata)
  ```

## 🚧 In Progress

### Granite Weight Loader
- **File**: `purego/tensor/generic_loader.go`
- **Status**: 🚧 Needs implementation
- **Issue**: Current loader expects GPT-2/Falcon/Llama tensor names
- **Needed**: Granite-specific `WeightMapping`

## Granite Tensor Structure

Based on analysis of the downloaded model:

### Token Embedding
```
model.embed_tokens.weight           [vocab_size, hidden] = [100352, 768]
```

### Mamba Layers (e.g., Layer 0)
```
model.layers.0.input_layernorm.weight              [hidden]
model.layers.0.mamba.A_log                         [num_heads, state_size]
model.layers.0.mamba.D                             [?]
model.layers.0.mamba.conv1d.weight                 [expand*hidden, 1, kernel]
model.layers.0.mamba.conv1d.bias                   [expand*hidden]
model.layers.0.mamba.dt_bias                       [num_heads]
model.layers.0.mamba.in_proj.weight                [2*expand*hidden, hidden]
model.layers.0.mamba.norm.weight                   [expand*hidden]
model.layers.0.mamba.out_proj.weight               [hidden, expand*hidden]
model.layers.0.post_attention_layernorm.weight     [hidden]
model.layers.0.shared_mlp.input_linear.weight      [intermediate, hidden]
model.layers.0.shared_mlp.output_linear.weight     [hidden, intermediate]
```

### Attention Layers (e.g., Layer 10)
```
model.layers.10.input_layernorm.weight              [hidden]
model.layers.10.self_attn.q_proj.weight             [hidden, hidden]
model.layers.10.self_attn.k_proj.weight             [kv_hidden, hidden]
model.layers.10.self_attn.v_proj.weight             [kv_hidden, hidden]
model.layers.10.self_attn.o_proj.weight             [hidden, hidden]
model.layers.10.post_attention_layernorm.weight     [hidden]
model.layers.10.shared_mlp.input_linear.weight      [intermediate, hidden]
model.layers.10.shared_mlp.output_linear.weight     [hidden, intermediate]
```

### Final Norm
```
model.norm.weight                   [hidden]
```

### Notes
- **No separate LM head**: Tied embeddings (`tie_word_embeddings: true`)
- **Shared MLP**: Both attention and Mamba layers have `shared_mlp`
- **Mamba-specific**: `A_log`, `D`, `dt_bias`, `conv1d`, `norm`
- **x_proj**: Missing in safetensors (might be computed from `in_proj`)

## Implementation Plan

### Step 1: Add Granite Weight Mapping

Create `GetGraniteMapping()` in `purego/tensor/generic_loader.go`:

```go
func GetGraniteMapping() *WeightMapping {
    return &WeightMapping{
        TokenEmbeddingKey:  "model.embed_tokens.weight",
        LayerPrefix:        "model.layers.{layer}",
        FinalNormKey:       "model.norm.weight",
        TiedEmbedding:      true,

        // Mamba-specific keys
        Mamba2ALogKey:      "mamba.A_log",
        Mamba2DKey:         "mamba.D",
        Mamba2Conv1dWeight: "mamba.conv1d.weight",
        Mamba2Conv1dBias:   "mamba.conv1d.bias",
        Mamba2DtBiasKey:    "mamba.dt_bias",
        Mamba2InProjKey:    "mamba.in_proj.weight",
        Mamba2NormKey:      "mamba.norm.weight",
        Mamba2OutProjKey:   "mamba.out_proj.weight",

        // Attention keys (GQA style)
        AttentionQKey:      "self_attn.q_proj.weight",
        AttentionKKey:      "self_attn.k_proj.weight",
        AttentionVKey:      "self_attn.v_proj.weight",
        AttentionOutKey:    "self_attn.o_proj.weight",

        // Shared MLP (for both layer types)
        FFNUpKey:           "shared_mlp.input_linear.weight",
        FFNDownKey:         "shared_mlp.output_linear.weight",

        // Norms
        InputNormKey:       "input_layernorm.weight",
        PostAttnNormKey:    "post_attention_layernorm.weight",
    }
}
```

### Step 2: Extend WeightMapping Structure

Add Mamba2-specific fields to `WeightMapping`:

```go
type WeightMapping struct {
    // Existing fields...

    // Mamba2 fields
    Mamba2ALogKey      string
    Mamba2DKey         string
    Mamba2Conv1dWeight string
    Mamba2Conv1dBias   string
    Mamba2DtBiasKey    string
    Mamba2InProjKey    string
    Mamba2XProjKey     string // For B, C, delta projection
    Mamba2DtProjKey    string // For delta projection
    Mamba2NormKey      string
    Mamba2OutProjKey   string
}
```

### Step 3: Implement Hybrid Layer Loading

Modify `LoadModel()` to handle hybrid architectures:

```go
func LoadModel(modelPath string, config *ModelConfig) (*TransformerModel, error) {
    // ... existing code ...

    // Load layers based on type
    for i := 0; i < config.NumLayers; i++ {
        layerType := "attention" // default
        if len(config.HybridLayers) > i {
            layerType = config.HybridLayers[i]
        }

        if layerType == "mamba2" {
            loadMamba2Layer(model.Blocks[i], tensors, mapping, i)
        } else {
            loadAttentionLayer(model.Blocks[i], tensors, mapping, i)
        }
    }

    return model, nil
}
```

### Step 4: Missing x_proj Handling

The Granite model doesn't have separate `x_proj` and `dt_proj` tensors. Need to investigate:

1. **Option A**: Computed from `in_proj`
   - `in_proj` might be split into multiple projections
   - Common in space-efficient models

2. **Option B**: Integrated into forward pass
   - The model computes projections on-the-fly
   - Need to trace through HuggingFace implementation

3. **Option C**: Different architecture variant
   - Granite might use a simplified Mamba2
   - Check model config for clues

**Action**: Review HuggingFace Granite implementation:
```bash
git clone https://github.com/huggingface/transformers
# Look at: src/transformers/models/granite/*
```

## Testing Plan

Once weight loader is complete:

### Test 1: Load Model
```bash
MODEL_DIR=./models/granite-350m go run ./purego/example_generic/main.go
```

Expected: Model loads without errors

### Test 2: Simple Generation
```bash
MODEL_DIR=./models/granite-350m ./bin/generic_test "What is 2+2?"
```

Expected: Reasonable response (might be wrong, but should generate)

### Test 3: Mamba2 Layer
```bash
MODEL_DIR=./models/granite-350m ./bin/generic_test "Explain Mamba2"
```

Expected: Multi-token generation using Mamba2 layers

### Test 4: Attention Layer
Verify attention layers (10, 13, 17, 27) are used correctly

## Current Error

```
Failed to load model: failed to load token embedding: required tensor 'wte.weight' not found
```

**Cause**: Weight loader uses GPT-2 naming convention
**Fix**: Implement Granite weight mapping (Step 1 above)

## Performance Targets

Once working, Granite should provide:

| Metric | Target | Basis |
|--------|--------|-------|
| Memory | ~500MB | 340M params × 4 bytes (fp32) |
| First token | < 500ms | Prefill phase |
| Generation | > 20 tokens/sec | Mamba2 linear complexity |
| Context | Up to 8K | Limited by config, model supports 32K |

## Resources

### Documentation
- `GRANITE_GUIDE.md` - User guide for Granite models
- `ARCHITECTURE_COMPARISON.md` - Architecture differences

### Code
- `purego/tensor/mamba2.go` - Mamba2 implementation
- `purego/tensor/config.go` - Granite configuration
- `purego/tensor/generic_model.go` - Hybrid model support
- `scripts/download_model.py` - Download script

### HuggingFace
- Model: https://huggingface.co/ibm-granite/granite-4.0-h-350m
- Config: https://huggingface.co/ibm-granite/granite-4.0-h-350m/blob/main/config.json
- Transformers: https://github.com/huggingface/transformers

### Papers
- Mamba: arXiv:2312.00752
- Mamba2: arXiv:2405.21060

## Next Actions

1. **Research x_proj**: Check HuggingFace Granite implementation
2. **Implement GetGraniteMapping()**: Add to generic_loader.go
3. **Extend WeightMapping**: Add Mamba2 fields
4. **Implement loadMamba2Layer()**: Load Mamba2 weights
5. **Test loading**: Verify all tensors load correctly
6. **Test generation**: Run inference tests
7. **Benchmark**: Measure memory and speed
8. **Document**: Update guides with results

## Conclusion

The foundation for Granite support is complete:
- ✅ Mamba2 layers implemented
- ✅ Hybrid architecture supported
- ✅ Model downloaded successfully
- 🚧 Weight loader needs Granite mapping
- 📋 Testing pending weight loader completion

Estimated remaining work: **2-4 hours** to implement weight loader and test.
