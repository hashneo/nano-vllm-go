package tensor

import "math"

// RoPE (Rotary Position Embedding) implementation
// Used by Falcon, Llama, Mistral, etc.

// RoPECache stores precomputed sin/cos values
type RoPECache struct {
	CosCache  *Tensor // [max_seq_len, head_dim]
	SinCache  *Tensor // [max_seq_len, head_dim]
	HeadDim   int
	MaxSeqLen int
	Base      float64 // Usually 10000.0
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
	// Following HuggingFace's approach: compute freqs for half dims, then duplicate
	// freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))  # [dim/2]
	// Then concatenate: [freq0, freq1, ..., freq31, freq0, freq1, ..., freq31]
	halfDim := headDim / 2
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < halfDim; i++ {
			// Compute frequency for this dimension (using 2*i for the exponent)
			freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq

			cosVal := float32(math.Cos(angle))
			sinVal := float32(math.Sin(angle))

			// Store in both first half and second half (concatenated pattern)
			cache.CosCache.Data[pos*headDim+i] = cosVal
			cache.CosCache.Data[pos*headDim+halfDim+i] = cosVal
			cache.SinCache.Data[pos*headDim+i] = sinVal
			cache.SinCache.Data[pos*headDim+halfDim+i] = sinVal
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

	// Check if K has different number of heads (MQA/GQA)
	numKVHeads := k.Shape[1] // Number of KV heads (1 for MQA, < numHeads for GQA)

	// Apply rotation to each position
	// Uses the "rotate_half" approach from HuggingFace transformers:
	// result = x * cos + rotate_half(x) * sin
	// where rotate_half splits head_dim in half and returns [-x2, x1]
	halfDim := headDim / 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := startPos + s
				if pos >= rc.MaxSeqLen {
					panic("Position exceeds max sequence length")
				}

				// Get indices for Q
				qIdx := b*numHeads*seqLen*headDim + h*seqLen*headDim + s*headDim

				// For K, map query head to KV head (for MQA/GQA)
				kvHead := h % numKVHeads // Map query head to corresponding KV head
				kIdx := b*numKVHeads*seqLen*headDim + kvHead*seqLen*headDim + s*headDim

				cacheIdx := pos * headDim

				// Store original values for Q and K
				qOriginal := make([]float32, headDim)
				kOriginal := make([]float32, headDim)
				copy(qOriginal, q.Data[qIdx:qIdx+headDim])
				copy(kOriginal, k.Data[kIdx:kIdx+headDim])

				// Apply rotation to Q: result = x * cos + rotate_half(x) * sin
				for i := 0; i < headDim; i++ {
					cos := rc.CosCache.Data[cacheIdx+i]
					sin := rc.SinCache.Data[cacheIdx+i]

					// rotate_half: first half gets -second_half, second half gets first_half
					var rotated float32
					if i < halfDim {
						rotated = -qOriginal[i+halfDim]
					} else {
						rotated = qOriginal[i-halfDim]
					}

					q.Data[qIdx+i] = qOriginal[i]*cos + rotated*sin
				}

				// Apply rotation to K: result = x * cos + rotate_half(x) * sin
				for i := 0; i < headDim; i++ {
					cos := rc.CosCache.Data[cacheIdx+i]
					sin := rc.SinCache.Data[cacheIdx+i]

					// rotate_half: first half gets -second_half, second half gets first_half
					var rotated float32
					if i < halfDim {
						rotated = -kOriginal[i+halfDim]
					} else {
						rotated = kOriginal[i-halfDim]
					}

					k.Data[kIdx+i] = kOriginal[i]*cos + rotated*sin
				}
			}
		}
	}
}

// ApplyRoPESingleTensor applies RoPE to a single tensor
// Useful for GQA where Q and K have different number of heads
// t: [batch, num_heads, seq, head_dim]
func (rc *RoPECache) ApplyRoPESingleTensor(t *Tensor, startPos int) {
	if len(t.Shape) != 4 {
		panic("RoPE expects 4D tensor [batch, num_heads, seq, head_dim]")
	}

	batch := t.Shape[0]
	numHeads := t.Shape[1]
	seqLen := t.Shape[2]
	headDim := t.Shape[3]

	if headDim != rc.HeadDim {
		panic("Head dimension mismatch")
	}

	// Apply rotation to each position
	// Uses the "rotate_half" approach from HuggingFace transformers:
	// result = x * cos + rotate_half(x) * sin
	// where rotate_half splits head_dim in half and returns [-x2, x1]
	halfDim := headDim / 2
	for b := 0; b < batch; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				pos := startPos + s
				if pos >= rc.MaxSeqLen {
					panic("Position exceeds max sequence length")
				}

				idx := b*numHeads*seqLen*headDim + h*seqLen*headDim + s*headDim
				cacheIdx := pos * headDim

				// Store original values
				original := make([]float32, headDim)
				copy(original, t.Data[idx:idx+headDim])

				// Apply: result = x * cos + rotate_half(x) * sin
				for i := 0; i < headDim; i++ {
					cos := rc.CosCache.Data[cacheIdx+i]
					sin := rc.SinCache.Data[cacheIdx+i]

					// rotate_half: first half gets -second_half, second half gets first_half
					var rotated float32
					if i < halfDim {
						rotated = -original[i+halfDim]
					} else {
						rotated = original[i-halfDim]
					}

					t.Data[idx+i] = original[i]*cos + rotated*sin
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
