# SwiGLU Bug Fix

## Summary

Fixed critical bug in SwiGLU activation function that caused Llama-based models (TinyLlama, Mistral-7B, Llama-3.2-1B) to generate repetitive garbage output.

## The Bug

In `purego/tensor/transformer.go`, the SwiGLU implementation had the gate and up projections **swapped**:

```go
// BEFORE (WRONG):
value := x.SliceLastDim(0, halfDim)        // First half
gate := x.SliceLastDim(halfDim, 2*halfDim) // Second half
result = value * silu(gate)  // Applied SiLU to SECOND half

// AFTER (CORRECT):
gate := x.SliceLastDim(0, halfDim)         // First half
up := x.SliceLastDim(halfDim, 2*halfDim)   // Second half
result = silu(gate) * up  // Apply SiLU to FIRST half
```

## Root Cause

The model loader concatenates weights as `[gateWeight | upWeight]`, so after `MatMul(x, W1)`:
- First half = gate_proj output (should have SiLU applied)
- Second half = up_proj output (no activation)

PyTorch SwiGLU formula: `down(silu(gate(x)) * up(x))`

Our code was doing: `down(gate(x) * silu(up(x)))` ❌

## Impact

**Before the fix:**
- All Llama models predicted punctuation (`,` `.` `-`) with high confidence
- Generated repetitive nonsense: "Angeles, Angeles, Angeles..."
- Top prediction after "What is the capital of Germany?": `,` (comma)

**After the fix:**
- Llama 3.2 correctly predicts: "The capital of Germany is Berlin."
- Top prediction: `The` → `capital` → `of` → `Germany` → `is` → `Berlin`

## Testing

Test with correct tokenization:
```bash
go run ./cmd/test-llama-generate/main.go
```

Expected output:
```
First token: 'The' (ID: 791)
Output: The capital of Germany is Berlin.
✓ Generation complete (hit EOS)
```

## Tokenizer Limitation

**Note:** The Go UniversalTokenizer currently has limited BPE support and may not tokenize Llama 3 prompts correctly (encodes 25 tokens instead of 17). For accurate tokenization:

1. Use the Python helper: `python3 scripts/encode_text.py <model_dir> "<text>"`
2. Or use pre-tokenized inputs as shown in `cmd/test-llama-generate/main.go`

Full BPE implementation with 280K+ merge rules is planned for future work.

## Files Modified

- `purego/tensor/transformer.go` - Fixed SwiGLU gate/up order (lines 50-64)
- `purego/universal_tokenizer.go` - Improved special token handling (partial fix)

## Verification

Compared against PyTorch reference implementation:
- ✅ Layer outputs match (within float32 precision)
- ✅ Attention patterns correct
- ✅ Logit predictions match
- ✅ Generation quality correct
