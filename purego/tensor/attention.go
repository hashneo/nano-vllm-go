package tensor

import (
	"fmt"
	"math"
)

// MultiHeadAttention implements multi-head self-attention
type MultiHeadAttention struct {
	NumHeads int
	HeadDim  int
	Hidden   int

	// Weights
	QWeight   *Tensor // [hidden, hidden]
	KWeight   *Tensor
	VWeight   *Tensor
	OutWeight *Tensor

	// Biases
	QBias   *Tensor // [hidden]
	KBias   *Tensor
	VBias   *Tensor
	OutBias *Tensor
}

// Forward performs multi-head attention
func (mha *MultiHeadAttention) Forward(x *Tensor) *Tensor {
	output, _, _ := mha.ForwardWithCache(x, nil, nil)
	return output
}

// ForwardWithCache performs multi-head attention with optional KV caching
// Returns: (output, new_k_cache, new_v_cache)
func (mha *MultiHeadAttention) ForwardWithCache(x *Tensor, kCache, vCache *Tensor) (*Tensor, *Tensor, *Tensor) {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	// Linear projections: x @ W + b
	Q := mha.project(x, mha.QWeight, mha.QBias)
	K := mha.project(x, mha.KWeight, mha.KBias)
	V := mha.project(x, mha.VWeight, mha.VBias)

	// Reshape to [batch, heads, seq, head_dim]
	Q = mha.splitHeads(Q, batchSize, seqLen)
	K = mha.splitHeads(K, batchSize, seqLen)
	V = mha.splitHeads(V, batchSize, seqLen)

	// If we have cache, concatenate along sequence dimension (dim=2)
	if kCache != nil && vCache != nil {
		K = Concatenate(kCache, K, 2)
		V = Concatenate(vCache, V, 2)
	}

	// Scaled dot-product attention (Q attends to full K,V including cache)
	output := mha.scaledDotProductAttention(Q, K, V)

	// Reshape back to [batch, seq, hidden]
	output = mha.combineHeads(output, batchSize, seqLen)

	// Output projection
	output = mha.project(output, mha.OutWeight, mha.OutBias)

	return output, K, V
}

func (mha *MultiHeadAttention) project(x, weight, bias *Tensor) *Tensor {
	// x: [batch, seq, hidden]
	// weight: [hidden, hidden]
	// Flatten batch and seq for matmul
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	xFlat := x.Reshape(batchSize*seqLen, mha.Hidden)
	result := MatMul(xFlat, weight)

	// Add bias
	if bias != nil {
		for i := 0; i < batchSize*seqLen; i++ {
			for j := 0; j < mha.Hidden; j++ {
				result.Data[i*mha.Hidden+j] += bias.Data[j]
			}
		}
	}

	return result.Reshape(batchSize, seqLen, mha.Hidden)
}

func (mha *MultiHeadAttention) splitHeads(x *Tensor, batchSize, seqLen int) *Tensor {
	// [batch, seq, hidden] -> [batch, seq, heads, head_dim]
	// -> [batch, heads, seq, head_dim]
	result := NewTensor(batchSize, mha.NumHeads, seqLen, mha.HeadDim)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < mha.NumHeads; h++ {
				for d := 0; d < mha.HeadDim; d++ {
					srcIdx := b*seqLen*mha.Hidden + s*mha.Hidden + h*mha.HeadDim + d
					dstIdx := b*mha.NumHeads*seqLen*mha.HeadDim + h*seqLen*mha.HeadDim + s*mha.HeadDim + d
					result.Data[dstIdx] = x.Data[srcIdx]
				}
			}
		}
	}

	return result
}

func (mha *MultiHeadAttention) combineHeads(x *Tensor, batchSize, seqLen int) *Tensor {
	// [batch, heads, seq, head_dim] -> [batch, seq, hidden]
	result := NewTensor(batchSize, seqLen, mha.Hidden)

	for b := 0; b < batchSize; b++ {
		for h := 0; h < mha.NumHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < mha.HeadDim; d++ {
					srcIdx := b*mha.NumHeads*seqLen*mha.HeadDim + h*seqLen*mha.HeadDim + s*mha.HeadDim + d
					dstIdx := b*seqLen*mha.Hidden + s*mha.Hidden + h*mha.HeadDim + d
					result.Data[dstIdx] = x.Data[srcIdx]
				}
			}
		}
	}

	return result
}

func (mha *MultiHeadAttention) scaledDotProductAttention(Q, K, V *Tensor) *Tensor {
	// Q: [batch, heads, q_seq, head_dim]
	// K, V: [batch, heads, kv_seq, head_dim]  (kv_seq may include cache)
	batchSize := Q.Shape[0]
	numHeads := Q.Shape[1]
	qSeqLen := Q.Shape[2]
	headDim := Q.Shape[3]
	kvSeqLen := K.Shape[2]

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	result := NewTensor(batchSize, numHeads, qSeqLen, headDim)

	// Process each batch and head separately
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			// Calculate offsets for this batch and head
			qOffset := b*numHeads*qSeqLen*headDim + h*qSeqLen*headDim
			kOffset := b*numHeads*kvSeqLen*headDim + h*kvSeqLen*headDim
			vOffset := b*numHeads*kvSeqLen*headDim + h*kvSeqLen*headDim

			// Compute attention scores: Q @ K^T
			scores := NewTensor(qSeqLen, kvSeqLen)
			for i := 0; i < qSeqLen; i++ {
				for j := 0; j < kvSeqLen; j++ {
					sum := float32(0)
					for d := 0; d < headDim; d++ {
						qVal := Q.Data[qOffset+i*headDim+d]
						kVal := K.Data[kOffset+j*headDim+d]
						sum += qVal * kVal
					}
					scores.Data[i*kvSeqLen+j] = sum * scale
				}
			}

			// Apply causal mask (for autoregressive generation)
			// Each query position i can only attend to key positions up to (kvSeqLen - qSeqLen + i)
			for i := 0; i < qSeqLen; i++ {
				maxPos := kvSeqLen - qSeqLen + i
				for j := maxPos + 1; j < kvSeqLen; j++ {
					scores.Data[i*kvSeqLen+j] = -1e10 // Mask future positions
				}
			}

			// Softmax
			scores = Softmax(scores)

			// Apply attention to values: scores @ V
			for i := 0; i < qSeqLen; i++ {
				for d := 0; d < headDim; d++ {
					sum := float32(0)
					for j := 0; j < kvSeqLen; j++ {
						attnWeight := scores.Data[i*kvSeqLen+j]
						vVal := V.Data[vOffset+j*headDim+d]
						sum += attnWeight * vVal
					}
					resultOffset := b*numHeads*qSeqLen*headDim + h*qSeqLen*headDim
					result.Data[resultOffset+i*headDim+d] = sum
				}
			}
		}
	}

	return result
}

// GroupedQueryAttention implements grouped-query attention (GQA)
type GroupedQueryAttention struct {
	NumHeads   int // Number of query heads
	NumKVHeads int // Number of KV heads (< NumHeads)
	HeadDim    int
	Hidden     int

	QWeight   *Tensor // [hidden, hidden]
	KWeight   *Tensor // [hidden, kv_hidden] where kv_hidden = num_kv_heads * head_dim
	VWeight   *Tensor // [hidden, kv_hidden]
	OutWeight *Tensor // [hidden, hidden]

	RoPECache *RoPECache // Rotary position embeddings (for Llama/Mistral)
}

// Forward performs GQA forward pass
func (gqa *GroupedQueryAttention) Forward(x *Tensor) *Tensor {
	output, _, _ := gqa.ForwardWithCache(x, nil, nil, 0)
	return output
}

// ForwardWithCache performs GQA with KV caching and position offset for RoPE
// Returns: (output, new_k_cache, new_v_cache)
func (gqa *GroupedQueryAttention) ForwardWithCache(x *Tensor, kCache, vCache *Tensor, posOffset int) (*Tensor, *Tensor, *Tensor) {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	// Project Q, K, V
	Q := gqa.projectQ(x, batchSize, seqLen)
	K := gqa.projectKV(x, gqa.KWeight, batchSize, seqLen)
	V := gqa.projectKV(x, gqa.VWeight, batchSize, seqLen)

	// Split into heads
	Q = gqa.splitHeadsQ(Q, batchSize, seqLen)
	K = gqa.splitHeadsKV(K, batchSize, seqLen)
	V = gqa.splitHeadsKV(V, batchSize, seqLen)

	// Apply RoPE if available (for Llama/Mistral)
	// Apply separately to Q and K since they have different number of heads in GQA
	// Use posOffset for correct positional information with KV caching
	if gqa.RoPECache != nil {
		gqa.RoPECache.ApplyRoPESingleTensor(Q, posOffset)
		gqa.RoPECache.ApplyRoPESingleTensor(K, posOffset)
	}

	// If we have cache, concatenate along sequence dimension (dim=2)
	// Cache has shape [batch, num_kv_heads, cached_seq, head_dim]
	if kCache != nil && vCache != nil {
		K = Concatenate(kCache, K, 2)
		V = Concatenate(vCache, V, 2)
	}

	// Save K and V before repeating (for cache storage)
	newKCache := K
	newVCache := V

	// Now repeat KV heads to match query heads
	// K, V now have full sequence length (cached + new)
	headsPerKV := gqa.NumHeads / gqa.NumKVHeads
	fullSeqLen := K.Shape[2]
	K = gqa.repeatKVHeads(K, headsPerKV, batchSize, fullSeqLen)
	V = gqa.repeatKVHeads(V, headsPerKV, batchSize, fullSeqLen)

	// Compute attention scores
	// Q has shape [batch, num_heads, q_seq, head_dim]
	// K, V have shape [batch, num_heads, kv_seq, head_dim] (may include cache)
	qSeqLen := seqLen  // New tokens being processed
	kvSeqLen := K.Shape[2]  // Full sequence including cache
	scores := gqa.computeScores(Q, K, batchSize, qSeqLen, kvSeqLen)
	attn := gqa.softmaxLastDim(scores, batchSize, qSeqLen, kvSeqLen)

	// Apply attention to V
	output := gqa.applyAttention(attn, V, batchSize, qSeqLen, kvSeqLen)

	// Merge heads and project output
	output = gqa.mergeHeads(output, batchSize, seqLen)
	output = gqa.projectOut(output, batchSize, seqLen)

	// Return unrepeated K,V (with num_kv_heads) for cache storage
	return output, newKCache, newVCache
}

// SetConfig allows GQA to access config (required by AttentionLayer interface)
func (gqa *GroupedQueryAttention) SetConfig(config *ModelConfig) {
	// RoPE cache is already initialized during model creation
}

func (gqa *GroupedQueryAttention) projectQ(x *Tensor, batchSize, seqLen int) *Tensor {
	xFlat := x.Reshape(batchSize*seqLen, gqa.Hidden)
	result := MatMul(xFlat, gqa.QWeight)
	return result.Reshape(batchSize, seqLen, gqa.Hidden)
}

func (gqa *GroupedQueryAttention) projectKV(x *Tensor, weight *Tensor, batchSize, seqLen int) *Tensor {
	kvHidden := gqa.NumKVHeads * gqa.HeadDim
	xFlat := x.Reshape(batchSize*seqLen, gqa.Hidden)
	result := MatMul(xFlat, weight)
	return result.Reshape(batchSize, seqLen, kvHidden)
}

func (gqa *GroupedQueryAttention) projectOut(x *Tensor, batchSize, seqLen int) *Tensor {
	xFlat := x.Reshape(batchSize*seqLen, gqa.Hidden)
	result := MatMul(xFlat, gqa.OutWeight)
	return result.Reshape(batchSize, seqLen, gqa.Hidden)
}

func (gqa *GroupedQueryAttention) splitHeadsQ(x *Tensor, batchSize, seqLen int) *Tensor {
	result := NewTensor(batchSize, gqa.NumHeads, seqLen, gqa.HeadDim)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < gqa.NumHeads; h++ {
				for d := 0; d < gqa.HeadDim; d++ {
					srcIdx := b*seqLen*gqa.Hidden + s*gqa.Hidden + h*gqa.HeadDim + d
					dstIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + s*gqa.HeadDim + d
					result.Data[dstIdx] = x.Data[srcIdx]
				}
			}
		}
	}
	return result
}

func (gqa *GroupedQueryAttention) splitHeadsKV(x *Tensor, batchSize, seqLen int) *Tensor {
	kvHidden := gqa.NumKVHeads * gqa.HeadDim
	result := NewTensor(batchSize, gqa.NumKVHeads, seqLen, gqa.HeadDim)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < gqa.NumKVHeads; h++ {
				for d := 0; d < gqa.HeadDim; d++ {
					srcIdx := b*seqLen*kvHidden + s*kvHidden + h*gqa.HeadDim + d
					dstIdx := b*gqa.NumKVHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + s*gqa.HeadDim + d
					result.Data[dstIdx] = x.Data[srcIdx]
				}
			}
		}
	}
	return result
}

func (gqa *GroupedQueryAttention) repeatKVHeads(x *Tensor, repeat int, batchSize, seqLen int) *Tensor {
	result := NewTensor(batchSize, gqa.NumHeads, seqLen, gqa.HeadDim)

	// Debug: Print for first decode call (first layer only)
	debugFirst := false  // Disabled for now
	if debugFirst && batchSize == 1 && seqLen == 1 {
		fmt.Printf("[repeatKVHeads] Input shape: %v, NumKVHeads=%d, NumHeads=%d, repeat=%d\n",
			x.Shape, gqa.NumKVHeads, gqa.NumHeads, repeat)
		fmt.Printf("[repeatKVHeads] Input KV head 0 first 3 values: %.4f, %.4f, %.4f\n",
			x.Data[0], x.Data[1], x.Data[2])
	}

	for b := 0; b < batchSize; b++ {
		for kvHead := 0; kvHead < gqa.NumKVHeads; kvHead++ {
			for r := 0; r < repeat; r++ {
				qHead := kvHead*repeat + r
				for s := 0; s < seqLen; s++ {
					for d := 0; d < gqa.HeadDim; d++ {
						srcIdx := b*gqa.NumKVHeads*seqLen*gqa.HeadDim + kvHead*seqLen*gqa.HeadDim + s*gqa.HeadDim + d
						dstIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + qHead*seqLen*gqa.HeadDim + s*gqa.HeadDim + d
						result.Data[dstIdx] = x.Data[srcIdx]
					}
				}
			}
		}
	}

	if debugFirst {
		fmt.Printf("[repeatKVHeads] Output Q head 0 (copy of KV head 0): %.4f, %.4f, %.4f\n",
			result.Data[0], result.Data[1], result.Data[2])
		// Check head 4 (should also be copy of KV head 0 if repeat=4)
		head4Offset := 4 * seqLen * gqa.HeadDim
		fmt.Printf("[repeatKVHeads] Output Q head 4 (should also be KV head 0 if repeat>=4): %.4f, %.4f, %.4f\n",
			result.Data[head4Offset], result.Data[head4Offset+1], result.Data[head4Offset+2])
	}

	return result
}

func (gqa *GroupedQueryAttention) computeScores(Q, K *Tensor, batchSize, qSeqLen, kvSeqLen int) *Tensor {
	// Q: [batch, num_heads, q_seq, head_dim]
	// K: [batch, num_heads, kv_seq, head_dim]
	// Output: [batch, num_heads, q_seq, kv_seq]
	scores := NewTensor(batchSize, gqa.NumHeads, qSeqLen, kvSeqLen)
	scale := float32(1.0) / float32(math.Sqrt(float64(gqa.HeadDim)))

	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < qSeqLen; i++ {
				for j := 0; j < kvSeqLen; j++ {
					// Apply causal mask: query position i can only attend to key positions <= i
					// When using KV cache, kvSeqLen includes cached tokens, so:
					// query position i can attend to cached positions [0, kvSeqLen-qSeqLen)
					// and current positions [kvSeqLen-qSeqLen, kvSeqLen-qSeqLen+i]
					maxAllowedPos := kvSeqLen - qSeqLen + i
					if j > maxAllowedPos {
						// Mask future positions
						scoresIdx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
						scores.Data[scoresIdx] = -1e10
						continue
					}

					sum := float32(0.0)
					for d := 0; d < gqa.HeadDim; d++ {
						qIdx := b*gqa.NumHeads*qSeqLen*gqa.HeadDim + h*qSeqLen*gqa.HeadDim + i*gqa.HeadDim + d
						kIdx := b*gqa.NumHeads*kvSeqLen*gqa.HeadDim + h*kvSeqLen*gqa.HeadDim + j*gqa.HeadDim + d
						sum += Q.Data[qIdx] * K.Data[kIdx]
					}

					scoresIdx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
					scores.Data[scoresIdx] = sum * scale
				}
			}
		}
	}

	return scores
}

func (gqa *GroupedQueryAttention) applyAttention(attn, V *Tensor, batchSize, qSeqLen, kvSeqLen int) *Tensor {
	// attn: [batch, num_heads, q_seq, kv_seq]
	// V: [batch, num_heads, kv_seq, head_dim]
	// Output: [batch, num_heads, q_seq, head_dim]
	output := NewTensor(batchSize, gqa.NumHeads, qSeqLen, gqa.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < qSeqLen; i++ {
				for d := 0; d < gqa.HeadDim; d++ {
					sum := float32(0.0)
					for j := 0; j < kvSeqLen; j++ {
						attnIdx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
						vIdx := b*gqa.NumHeads*kvSeqLen*gqa.HeadDim + h*kvSeqLen*gqa.HeadDim + j*gqa.HeadDim + d
						sum += attn.Data[attnIdx] * V.Data[vIdx]
					}
					outIdx := b*gqa.NumHeads*qSeqLen*gqa.HeadDim + h*qSeqLen*gqa.HeadDim + i*gqa.HeadDim + d
					output.Data[outIdx] = sum
				}
			}
		}
	}
	return output
}

func (gqa *GroupedQueryAttention) mergeHeads(x *Tensor, batchSize, seqLen int) *Tensor {
	result := NewTensor(batchSize, seqLen, gqa.Hidden)
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLen; s++ {
			for h := 0; h < gqa.NumHeads; h++ {
				for d := 0; d < gqa.HeadDim; d++ {
					srcIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + s*gqa.HeadDim + d
					dstIdx := b*seqLen*gqa.Hidden + s*gqa.Hidden + h*gqa.HeadDim + d
					result.Data[dstIdx] = x.Data[srcIdx]
				}
			}
		}
	}
	return result
}

func (gqa *GroupedQueryAttention) softmaxLastDim(x *Tensor, batchSize, qSeqLen, kvSeqLen int) *Tensor {
	// Apply softmax over the last dimension (kv_seq) for each query position
	// Input: [batch, num_heads, q_seq, kv_seq]
	result := NewTensor(x.Shape...)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < qSeqLen; i++ {
				// Find max for numerical stability
				maxVal := float32(-1e10)
				for j := 0; j < kvSeqLen; j++ {
					idx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
					if x.Data[idx] > maxVal {
						maxVal = x.Data[idx]
					}
				}
				// Compute exp and sum
				sum := float32(0.0)
				for j := 0; j < kvSeqLen; j++ {
					idx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
					result.Data[idx] = float32(math.Exp(float64(x.Data[idx] - maxVal)))
					sum += result.Data[idx]
				}
				// Normalize
				for j := 0; j < kvSeqLen; j++ {
					idx := b*gqa.NumHeads*qSeqLen*kvSeqLen + h*qSeqLen*kvSeqLen + i*kvSeqLen + j
					result.Data[idx] /= sum
				}
			}
		}
	}
	return result
}
