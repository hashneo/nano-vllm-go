package tensor

import "math"

// MultiQueryAttention implements attention with shared K,V across all heads
// Used by Falcon models
// MQA: 71 query heads but only 1 KV head (extreme memory savings!)
type MultiQueryAttention struct {
	NumHeads int // Number of query heads (e.g., 71 for Falcon 7B)
	HeadDim  int // Dimension per head (e.g., 64)
	Hidden   int // Total hidden dimension (num_heads * head_dim)

	// Weights
	QWeight   *Tensor // [hidden, hidden] - separate queries for each head
	KVWeight  *Tensor // [hidden, head_dim * 2] - single K,V shared by all heads
	OutWeight *Tensor // [hidden, hidden]

	// Biases (Falcon doesn't use these, but keeping for flexibility)
	QBias   *Tensor
	KVBias  *Tensor
	OutBias *Tensor

	// RoPE cache
	RoPE *RoPECache
}

// NewMultiQueryAttention creates MQA layer
func NewMultiQueryAttention(numHeads, headDim int, maxSeqLen int) *MultiQueryAttention {
	hidden := numHeads * headDim

	return &MultiQueryAttention{
		NumHeads: numHeads,
		HeadDim:  headDim,
		Hidden:   hidden,
		RoPE:     NewRoPECache(headDim, maxSeqLen, 10000.0),
	}
}

// Forward performs multi-query attention without caching
func (mqa *MultiQueryAttention) Forward(x *Tensor) *Tensor {
	output, _, _ := mqa.ForwardWithCache(x, nil, nil, 0)
	return output
}

// ForwardWithCache performs MQA with KV caching and position offset for RoPE
// Returns: (output, new_k_cache, new_v_cache)
func (mqa *MultiQueryAttention) ForwardWithCache(x *Tensor, kCache, vCache *Tensor, posOffset int) (*Tensor, *Tensor, *Tensor) {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	// Project to queries (one per head)
	Q := mqa.projectQ(x, batchSize, seqLen)

	// Project to key and value (shared across all heads)
	K, V := mqa.projectKV(x, batchSize, seqLen)

	// Reshape Q to [batch, num_heads, seq, head_dim]
	Q = mqa.reshapeQ(Q, batchSize, seqLen)

	// K,V are [batch, 1, seq, head_dim]
	K = K.Reshape(batchSize, 1, seqLen, mqa.HeadDim)
	V = V.Reshape(batchSize, 1, seqLen, mqa.HeadDim)

	// Apply RoPE with correct position offset
	if mqa.RoPE != nil {
		mqa.RoPE.ApplyRoPE(Q, K, posOffset)
	}

	// If we have cache, concatenate along sequence dimension (dim=2)
	if kCache != nil && vCache != nil {
		K = Concatenate(kCache, K, 2)
		V = Concatenate(vCache, V, 2)
	}

	// Save K and V for cache (before repeating)
	newKCache := K
	newVCache := V

	// Repeat single KV head to match all query heads
	// K, V: [batch, 1, full_seq, head_dim] -> [batch, num_heads, full_seq, head_dim]
	fullSeqLen := K.Shape[2]
	K = mqa.repeatKVHeads(K, batchSize, fullSeqLen)
	V = mqa.repeatKVHeads(V, batchSize, fullSeqLen)

	// Scaled dot-product attention
	output := mqa.scaledDotProductMQA(Q, K, V, batchSize, seqLen)

	// Transpose from [batch, num_heads, seq, head_dim] to [batch, seq, num_heads, head_dim]
	// This is required before we can flatten to [batch, seq, hidden]!
	output = mqa.transposeHeadsAndSeq(output, batchSize, seqLen)

	// Now flatten to [batch, seq, hidden]
	output = output.Reshape(batchSize, seqLen, mqa.Hidden)

	// Output projection
	output = mqa.projectOut(output, batchSize, seqLen)

	return output, newKCache, newVCache
}

func (mqa *MultiQueryAttention) projectQ(x *Tensor, batchSize, seqLen int) *Tensor {
	xFlat := x.Reshape(batchSize*seqLen, mqa.Hidden)
	Q := MatMul(xFlat, mqa.QWeight)

	if mqa.QBias != nil {
		for i := 0; i < batchSize*seqLen; i++ {
			for j := 0; j < mqa.Hidden; j++ {
				Q.Data[i*mqa.Hidden+j] += mqa.QBias.Data[j]
			}
		}
	}

	return Q.Reshape(batchSize, seqLen, mqa.Hidden)
}

func (mqa *MultiQueryAttention) projectKV(x *Tensor, batchSize, seqLen int) (*Tensor, *Tensor) {
	xFlat := x.Reshape(batchSize*seqLen, mqa.Hidden)
	KV := MatMul(xFlat, mqa.KVWeight) // [batch*seq, head_dim*2]

	if mqa.KVBias != nil {
		for i := 0; i < batchSize*seqLen; i++ {
			for j := 0; j < mqa.HeadDim*2; j++ {
				KV.Data[i*mqa.HeadDim*2+j] += mqa.KVBias.Data[j]
			}
		}
	}

	// Split into K and V
	K := NewTensor(batchSize, seqLen, mqa.HeadDim)
	V := NewTensor(batchSize, seqLen, mqa.HeadDim)

	for i := 0; i < batchSize*seqLen; i++ {
		for j := 0; j < mqa.HeadDim; j++ {
			K.Data[i*mqa.HeadDim+j] = KV.Data[i*mqa.HeadDim*2+j]
			V.Data[i*mqa.HeadDim+j] = KV.Data[i*mqa.HeadDim*2+mqa.HeadDim+j]
		}
	}

	return K, V
}

func (mqa *MultiQueryAttention) reshapeQ(Q *Tensor, batchSize, seqLen int) *Tensor {
	result := NewTensor(batchSize, mqa.NumHeads, seqLen, mqa.HeadDim)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < mqa.NumHeads; h++ {
				for d := 0; d < mqa.HeadDim; d++ {
					srcIdx := b*seqLen*mqa.Hidden + s*mqa.Hidden + h*mqa.HeadDim + d
					dstIdx := b*mqa.NumHeads*seqLen*mqa.HeadDim + h*seqLen*mqa.HeadDim + s*mqa.HeadDim + d
					result.Data[dstIdx] = Q.Data[srcIdx]
				}
			}
		}
	}

	return result
}

// repeatKVHeads repeats the single KV head to match all query heads
// Input: [batch, 1, seq, head_dim]
// Output: [batch, num_heads, seq, head_dim]
func (mqa *MultiQueryAttention) repeatKVHeads(kv *Tensor, batchSize, seqLen int) *Tensor {
	result := NewTensor(batchSize, mqa.NumHeads, seqLen, mqa.HeadDim)

	// Copy the single KV head to all query head positions
	for b := 0; b < batchSize; b++ {
		for h := 0; h < mqa.NumHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < mqa.HeadDim; d++ {
					// Source: single head (h=0)
					srcIdx := b*1*seqLen*mqa.HeadDim + 0*seqLen*mqa.HeadDim + s*mqa.HeadDim + d
					// Destination: current query head
					dstIdx := b*mqa.NumHeads*seqLen*mqa.HeadDim + h*seqLen*mqa.HeadDim + s*mqa.HeadDim + d
					result.Data[dstIdx] = kv.Data[srcIdx]
				}
			}
		}
	}

	return result
}

func (mqa *MultiQueryAttention) scaledDotProductMQA(Q, K, V *Tensor, batchSize, qSeqLen int) *Tensor {
	scale := float32(1.0 / math.Sqrt(float64(mqa.HeadDim)))
	result := NewTensor(batchSize, mqa.NumHeads, qSeqLen, mqa.HeadDim)

	// K and V now have shape [batch, num_heads, kv_seq, head_dim] (after repeating)
	// where kv_seq >= qSeqLen (includes cached keys/values)
	kvSeqLen := K.Shape[2]

	// Each query head attends to its corresponding repeated K,V
	for b := 0; b < batchSize; b++ {
		for h := 0; h < mqa.NumHeads; h++ {
			// Compute attention scores: Q @ K^T
			scores := NewTensor(qSeqLen, kvSeqLen)

			for i := 0; i < qSeqLen; i++ {
				for j := 0; j < kvSeqLen; j++ {
					sum := float32(0)
					for d := 0; d < mqa.HeadDim; d++ {
						qIdx := b*mqa.NumHeads*qSeqLen*mqa.HeadDim + h*qSeqLen*mqa.HeadDim + i*mqa.HeadDim + d
						kIdx := b*mqa.NumHeads*kvSeqLen*mqa.HeadDim + h*kvSeqLen*mqa.HeadDim + j*mqa.HeadDim + d
						sum += Q.Data[qIdx] * K.Data[kIdx]
					}
					scores.Data[i*kvSeqLen+j] = sum * scale
				}
			}

			// Apply causal mask
			// With KV cache: kvSeqLen = cached_len + q_len
			// Query at position i (relative to current batch) can attend to:
			// - All cached positions (0 to cached_len-1)
			// - Current positions up to i (cached_len to cached_len+i)
			// So mask positions > cached_len + i
			cachedLen := kvSeqLen - qSeqLen
			for i := 0; i < qSeqLen; i++ {
				// Mask future positions beyond current query position
				for j := cachedLen + i + 1; j < kvSeqLen; j++ {
					scores.Data[i*kvSeqLen+j] = -1e10
				}
			}

			// Softmax over key dimension
			scores = Softmax(scores)

			// Weighted sum of values: scores @ V
			for i := 0; i < qSeqLen; i++ {
				for d := 0; d < mqa.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < kvSeqLen; j++ {
						vIdx := b*mqa.NumHeads*kvSeqLen*mqa.HeadDim + h*kvSeqLen*mqa.HeadDim + j*mqa.HeadDim + d
						sum += scores.Data[i*kvSeqLen+j] * V.Data[vIdx]
					}
					resultIdx := b*mqa.NumHeads*qSeqLen*mqa.HeadDim + h*qSeqLen*mqa.HeadDim + i*mqa.HeadDim + d
					result.Data[resultIdx] = sum
				}
			}
		}
	}

	return result
}

// transposeHeadsAndSeq transposes from [batch, num_heads, seq, head_dim] to [batch, seq, num_heads, head_dim]
// This is required before flattening to [batch, seq, hidden] for output projection
func (mqa *MultiQueryAttention) transposeHeadsAndSeq(input *Tensor, batchSize, seqLen int) *Tensor {
	// Input:  [batch, num_heads, seq, head_dim]
	// Output: [batch, seq, num_heads, head_dim]
	result := NewTensor(batchSize, seqLen, mqa.NumHeads, mqa.HeadDim)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < mqa.NumHeads; h++ {
				for d := 0; d < mqa.HeadDim; d++ {
					// Source: [batch, num_heads, seq, head_dim]
					srcIdx := b*mqa.NumHeads*seqLen*mqa.HeadDim + h*seqLen*mqa.HeadDim + s*mqa.HeadDim + d
					// Destination: [batch, seq, num_heads, head_dim]
					dstIdx := b*seqLen*mqa.NumHeads*mqa.HeadDim + s*mqa.NumHeads*mqa.HeadDim + h*mqa.HeadDim + d
					result.Data[dstIdx] = input.Data[srcIdx]
				}
			}
		}
	}

	return result
}

func (mqa *MultiQueryAttention) projectOut(x *Tensor, batchSize, seqLen int) *Tensor {
	xFlat := x.Reshape(batchSize*seqLen, mqa.Hidden)
	output := MatMul(xFlat, mqa.OutWeight)

	if mqa.OutBias != nil {
		for i := 0; i < batchSize*seqLen; i++ {
			for j := 0; j < mqa.Hidden; j++ {
				output.Data[i*mqa.Hidden+j] += mqa.OutBias.Data[j]
			}
		}
	}

	return output.Reshape(batchSize, seqLen, mqa.Hidden)
}
