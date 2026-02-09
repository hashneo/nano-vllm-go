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
	QWeight *Tensor // [hidden, hidden]
	KWeight *Tensor
	VWeight *Tensor
	OutWeight *Tensor

	// Biases
	QBias *Tensor // [hidden]
	KBias *Tensor
	VBias *Tensor
	OutBias *Tensor
}

// Forward performs multi-head attention
func (mha *MultiHeadAttention) Forward(x *Tensor) *Tensor {
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

	// Scaled dot-product attention
	output := mha.scaledDotProductAttention(Q, K, V)

	// Reshape back to [batch, seq, hidden]
	output = mha.combineHeads(output, batchSize, seqLen)

	// Output projection
	output = mha.project(output, mha.OutWeight, mha.OutBias)

	return output
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
