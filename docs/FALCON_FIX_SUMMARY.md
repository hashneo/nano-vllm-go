# Falcon Model Fix Summary

## Issue: Garbage Output from Falcon-7B

The Falcon model was producing garbage output because of an incorrect implementation of the QKV weight matrix splitting function.

## Root Cause

The `splitFalconQKV` function in `purego/tensor/generic_loader.go` was incorrectly assuming a **contiguous block layout** for the combined QKV weight matrix:

```
[All Q weights] [All K weights] [All V weights]
```

However, HuggingFace's Falcon implementation uses an **interleaved per-row layout**:

```
For each input feature (row):
[Q_head0] [Q_head1] ... [Q_head70] [K] [V]
```

### Technical Details

For Falcon-7B with Multi-Query Attention (MQA):
- **71 query heads**, **1 key head**, **1 value head**
- **Head dimension**: 64
- **Hidden size**: 4544 (= 71 × 64)
- **QKV weight matrix shape**: `[4544, 4672]` where 4672 = (71 + 2) × 64

Each row of the weight matrix contains:
- 71 chunks of 64 elements for Q heads (indices 0-70)
- 1 chunk of 64 elements for K (index 71)
- 1 chunk of 64 elements for V (index 72)

The old code was reading:
- Q: columns 0-4543 ✗ (incorrect - contains mixed Q/K/V data)
- K: columns 4544-4607 ✗ (incorrect - random portion of interleaved data)
- V: columns 4608-4671 ✗ (incorrect - random portion of interleaved data)

## The Fix

Updated `splitFalconQKV` function to properly **deinterleave** the weights:

```go
// For each row in the weight matrix
for row := 0; row < hidden; row++ {
    srcRowOffset := row * (numHeads + 2) * headDim

    // Extract all Q heads (interleaved chunks)
    for h := 0; h < numHeads; h++ {
        srcOffset := srcRowOffset + h*headDim
        dstOffset := row*numHeads*headDim + h*headDim
        copy(Q.Data[dstOffset:dstOffset+headDim],
             qkvWeight.Data[srcOffset:srcOffset+headDim])
    }

    // Extract K (at position numHeads * headDim)
    srcK := srcRowOffset + numHeads*headDim
    copy(K.Data[row*headDim:row*headDim+headDim],
         qkvWeight.Data[srcK:srcK+headDim])

    // Extract V (at position (numHeads+1) * headDim)
    srcV := srcRowOffset + (numHeads+1)*headDim
    copy(V.Data[row*headDim:row*headDim+headDim],
         qkvWeight.Data[srcV:srcV+headDim])
}
```

## Verification

Created comprehensive tests in `purego/tensor/falcon_split_test.go`:
- ✅ `TestSplitFalconQKV`: Tests with simplified dimensions (3 heads)
- ✅ `TestSplitFalconQKV_RealFalcon7BDimensions`: Tests with actual Falcon-7B dimensions

Both tests pass successfully.

## Additional Notes

### Multi-Query Attention (MQA) Configuration

The config loader correctly handles Falcon's MQA configuration:
- Reads `"multi_query": true` from config.json
- Overrides `num_kv_heads` to 1 (even if config has a different value)
- This ensures the single K/V head is shared across all 71 query heads

### RoPE Implementation

The RoPE implementation was already correct:
- Uses HuggingFace's "rotate_half" method
- Properly handles the MQA case (applies to 1 KV head, 71 query heads)
- Default base frequency of 10000.0 (Falcon doesn't specify rope_theta in config)

### Parallel Attention Architecture

Falcon uses parallel attention + FFN (both computed in parallel and added to residual):
- Already correctly implemented in `purego/tensor/falcon.go`
- Single input LayerNorm feeds both attention and FFN
- Both outputs are added to the residual connection

## Expected Behavior After Fix

With the corrected weight splitting, Falcon-7B should now:
1. Load the QKV weights correctly with proper head separation
2. Apply MQA correctly (71 query heads attending to 1 shared K/V head)
3. Generate coherent output instead of garbage

### Testing the Fix

To test with the actual model:

```bash
make ask
./bin/ask falcon "What is the capital of Germany?"
```

Expected output: "Berlin" (or similar coherent response)

## References

- **HuggingFace Transformers**: Falcon attention implementation splits QKV with shape `[batch, seq, num_heads + 2, head_dim]`
- **Research findings**: 10-100x memory reduction with MQA vs standard MHA
- **Original paper**: Falcon uses parallel attention for faster training/inference
