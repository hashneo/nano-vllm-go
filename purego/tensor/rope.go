package tensor

import "math"

// RoPE (Rotary Position Embedding) implementation
// Used by Falcon, Llama, Mistral, etc.

// RoPECache stores precomputed sin/cos values
type RoPECache struct {
	CosCache *Tensor // [max_seq_len, head_dim]
	SinCache *Tensor // [max_seq_len, head_dim]
	HeadDim  int
	MaxSeqLen int
	Base     float64 // Usually 10000.0
}

// NewRoPECache creates a cache of rotary embeddings
func NewRoPECache(headDim, maxSeqLen int, base float64) *RoPECache {
	cache := &RoPECache{
		HeadDim:   headDim,
		MaxSeqLen: maxSeqLen,
		Base:      base,
		CosCache:  NewTensor(maxSeqLen, headDim),
		SinCache:  NewTensor(maxSeqLen, headDim),
	}

	// Precompute cos and sin for all positions
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < headDim/2; i++ {
			// Compute frequency for this dimension pair
			freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq

			// Store cos and sin (each applies to two dimensions)
			cache.CosCache.Data[pos*headDim+2*i] = float32(math.Cos(angle))
			cache.CosCache.Data[pos*headDim+2*i+1] = float32(math.Cos(angle))
			cache.SinCache.Data[pos*headDim+2*i] = float32(math.Sin(angle))
			cache.SinCache.Data[pos*headDim+2*i+1] = float32(math.Sin(angle))
		}
	}

	return cache
}

// ApplyRoPE applies rotary position embeddings to query and key tensors
// q, k: [batch, seq, num_heads, head_dim] or [batch, num_heads, seq, head_dim]
// startPos: starting position in sequence (for KV cache)
func (rc *RoPECache) ApplyRoPE(q, k *Tensor, startPos int) {
	// Assume format: [batch, num_heads, seq, head_dim]
	if len(q.Shape) != 4 {
		panic("RoPE expects 4D tensor [batch, num_heads, seq, head_dim]")
	}

	batch := q.Shape[0]
	numHeads := q.Shape[1]
	seqLen := q.Shape[2]
	headDim := q.Shape[3]

	if headDim != rc.HeadDim {
		panic("Head dimension mismatch")
	}

	// Apply rotation to each position
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := startPos + s
				if pos >= rc.MaxSeqLen {
					panic("Position exceeds max sequence length")
				}

				// Rotate pairs of dimensions
				for i := 0; i < headDim/2; i++ {
					// Get indices
					qIdx := b*numHeads*seqLen*headDim + h*seqLen*headDim + s*headDim
					kIdx := qIdx
					cacheIdx := pos * headDim

					// Get values
					q0 := q.Data[qIdx+2*i]
					q1 := q.Data[qIdx+2*i+1]
					k0 := k.Data[kIdx+2*i]
					k1 := k.Data[kIdx+2*i+1]

					cos := rc.CosCache.Data[cacheIdx+2*i]
					sin := rc.SinCache.Data[cacheIdx+2*i]

					// Rotate query
					q.Data[qIdx+2*i] = q0*cos - q1*sin
					q.Data[qIdx+2*i+1] = q0*sin + q1*cos

					// Rotate key
					k.Data[kIdx+2*i] = k0*cos - k1*sin
					k.Data[kIdx+2*i+1] = k0*sin + k1*cos
				}
			}
		}
	}
}

// ApplyRoPEInplace is an optimized version that modifies tensors in place
func ApplyRoPEInplace(q *Tensor, cosCache, sinCache []float32, position int, numHeads, headDim int) {
	// Simple version for single position
	offset := position * headDim

	for h := 0; h < numHeads; h++ {
		headOffset := h * headDim

		for i := 0; i < headDim/2; i++ {
			idx := headOffset + 2*i
			cacheIdx := offset + 2*i

			q0 := q.Data[idx]
			q1 := q.Data[idx+1]

			cos := cosCache[cacheIdx]
			sin := sinCache[cacheIdx]

			q.Data[idx] = q0*cos - q1*sin
			q.Data[idx+1] = q0*sin + q1*cos
		}
	}
}
