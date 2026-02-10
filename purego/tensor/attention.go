package tensor

import (
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
	// Q, K, V: [batch, heads, seq, head_dim]
	batchSize := Q.Shape[0]
	numHeads := Q.Shape[1]
	seqLen := Q.Shape[2]
	headDim := Q.Shape[3]

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	result := NewTensor(batchSize, numHeads, seqLen, headDim)

	// Process each batch and head separately
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			// Extract Q, K, V for this batch and head
			qOffset := b*numHeads*seqLen*headDim + h*seqLen*headDim
			kOffset := qOffset
			vOffset := qOffset

			// Compute attention scores: Q @ K^T
			scores := NewTensor(seqLen, seqLen)
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0)
					for d := 0; d < headDim; d++ {
						qVal := Q.Data[qOffset+i*headDim+d]
						kVal := K.Data[kOffset+j*headDim+d]
						sum += qVal * kVal
					}
					scores.Data[i*seqLen+j] = sum * scale
				}
			}

			// Apply causal mask (for autoregressive generation)
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < seqLen; j++ {
					scores.Data[i*seqLen+j] = -1e10 // Mask future positions
				}
			}

			// Softmax
			scores = Softmax(scores)

			// Apply attention to values: scores @ V
			for i := 0; i < seqLen; i++ {
				for d := 0; d < headDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						attnWeight := scores.Data[i*seqLen+j]
						vVal := V.Data[vOffset+j*headDim+d]
						sum += attnWeight * vVal
					}
					resultOffset := b*numHeads*seqLen*headDim + h*seqLen*headDim
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
}

// Forward performs GQA forward pass
func (gqa *GroupedQueryAttention) Forward(x *Tensor) *Tensor {
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

	// Repeat KV heads to match query heads
	headsPerKV := gqa.NumHeads / gqa.NumKVHeads
	K = gqa.repeatKVHeads(K, headsPerKV, batchSize, seqLen)
	V = gqa.repeatKVHeads(V, headsPerKV, batchSize, seqLen)

	// Compute attention scores
	scores := gqa.computeScores(Q, K, batchSize, seqLen)
	attn := gqa.softmaxLastDim(scores, batchSize, seqLen)

	// Apply attention to V
	output := gqa.applyAttention(attn, V, batchSize, seqLen)

	// Merge heads and project output
	output = gqa.mergeHeads(output, batchSize, seqLen)
	output = gqa.projectOut(output, batchSize, seqLen)

	return output
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
	return result
}

func (gqa *GroupedQueryAttention) computeScores(Q, K *Tensor, batchSize, seqLen int) *Tensor {
	scores := NewTensor(batchSize, gqa.NumHeads, seqLen, seqLen)
	scale := float32(1.0) / float32(gqa.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0.0)
					for d := 0; d < gqa.HeadDim; d++ {
						qIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + i*gqa.HeadDim + d
						kIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + j*gqa.HeadDim + d
						sum += Q.Data[qIdx] * K.Data[kIdx]
					}
					scoresIdx := b*gqa.NumHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					scores.Data[scoresIdx] = sum * scale
				}
			}
		}
	}
	return scores
}

func (gqa *GroupedQueryAttention) applyAttention(attn, V *Tensor, batchSize, seqLen int) *Tensor {
	output := NewTensor(batchSize, gqa.NumHeads, seqLen, gqa.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < gqa.HeadDim; d++ {
					sum := float32(0.0)
					for j := 0; j < seqLen; j++ {
						attnIdx := b*gqa.NumHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
						vIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + j*gqa.HeadDim + d
						sum += attn.Data[attnIdx] * V.Data[vIdx]
					}
					outIdx := b*gqa.NumHeads*seqLen*gqa.HeadDim + h*seqLen*gqa.HeadDim + i*gqa.HeadDim + d
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

func (gqa *GroupedQueryAttention) SetConfig(config *ModelConfig) {
	// GQA doesn't need RoPE for Granite (uses NoPE)
}

func (gqa *GroupedQueryAttention) softmaxLastDim(x *Tensor, batchSize, seqLen int) *Tensor {
	// Apply softmax over the last dimension (for each query position)
	result := NewTensor(x.Shape...)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < gqa.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				// Find max for numerical stability
				maxVal := float32(-1e10)
				for j := 0; j < seqLen; j++ {
					idx := b*gqa.NumHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					if x.Data[idx] > maxVal {
						maxVal = x.Data[idx]
					}
				}
				// Compute exp and sum
				sum := float32(0.0)
				for j := 0; j < seqLen; j++ {
					idx := b*gqa.NumHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					result.Data[idx] = float32(math.Exp(float64(x.Data[idx] - maxVal)))
					sum += result.Data[idx]
				}
				// Normalize
				for j := 0; j < seqLen; j++ {
					idx := b*gqa.NumHeads*seqLen*seqLen + h*seqLen*seqLen + i*seqLen + j
					result.Data[idx] /= sum
				}
			}
		}
	}
	return result
}
