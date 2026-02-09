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

// Forward performs multi-query attention
func (mqa *MultiQueryAttention) Forward(x *Tensor) *Tensor {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	// Project to queries (one per head)
	Q := mqa.projectQ(x, batchSize, seqLen)

	// Project to key and value (shared across all heads)
	K, V := mqa.projectKV(x, batchSize, seqLen)

	// Reshape Q to [batch, num_heads, seq, head_dim]
	Q = mqa.reshapeQ(Q, batchSize, seqLen)

	// K,V are already [batch, 1, seq, head_dim]
	K = K.Reshape(batchSize, 1, seqLen, mqa.HeadDim)
	V = V.Reshape(batchSize, 1, seqLen, mqa.HeadDim)

	// Apply RoPE to Q and K
	mqa.RoPE.ApplyRoPE(Q, K, 0)

	// Scaled dot-product attention (Q uses shared K,V)
	output := mqa.scaledDotProductMQA(Q, K, V, batchSize, seqLen)

	// Reshape back to [batch, seq, hidden]
	output = output.Reshape(batchSize, seqLen, mqa.Hidden)

	// Output projection
	output = mqa.projectOut(output, batchSize, seqLen)

	return output
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

func (mqa *MultiQueryAttention) scaledDotProductMQA(Q, K, V *Tensor, batchSize, seqLen int) *Tensor {
	scale := float32(1.0 / math.Sqrt(float64(mqa.HeadDim)))
	result := NewTensor(batchSize, mqa.NumHeads, seqLen, mqa.HeadDim)

	// Each query head attends to the single shared K,V
	for b := 0; b < batchSize; b++ {
		for h := 0; h < mqa.NumHeads; h++ {
			// Compute attention scores for this head
			scores := NewTensor(seqLen, seqLen)

			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0)
					for d := 0; d < mqa.HeadDim; d++ {
						qIdx := b*mqa.NumHeads*seqLen*mqa.HeadDim + h*seqLen*mqa.HeadDim + i*mqa.HeadDim + d
						// K has only 1 head (index 0)
						kIdx := b*1*seqLen*mqa.HeadDim + 0*seqLen*mqa.HeadDim + j*mqa.HeadDim + d
						sum += Q.Data[qIdx] * K.Data[kIdx]
					}
					scores.Data[i*seqLen+j] = sum * scale
				}
			}

			// Apply causal mask
			for i := 0; i < seqLen; i++ {
				for j := i + 1; j < seqLen; j++ {
					scores.Data[i*seqLen+j] = -1e10
				}
			}

			// Softmax
			scores = Softmax(scores)

			// Apply to values
			for i := 0; i < seqLen; i++ {
				for d := 0; d < mqa.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						// V has only 1 head (index 0)
						vIdx := b*1*seqLen*mqa.HeadDim + 0*seqLen*mqa.HeadDim + j*mqa.HeadDim + d
						sum += scores.Data[i*seqLen+j] * V.Data[vIdx]
					}
					resultIdx := b*mqa.NumHeads*seqLen*mqa.HeadDim + h*seqLen*mqa.HeadDim + i*mqa.HeadDim + d
					result.Data[resultIdx] = sum
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
