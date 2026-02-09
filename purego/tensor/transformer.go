package tensor

// TransformerBlock implements a single transformer layer
type TransformerBlock struct {
	Attention *MultiHeadAttention
	FFN       *FeedForward
	LN1       *LayerNormLayer
	LN2       *LayerNormLayer
}

// Forward applies the transformer block
func (block *TransformerBlock) Forward(x *Tensor) *Tensor {
	// Self-attention with residual connection
	residual := x
	x = block.LN1.Forward(x)
	x = block.Attention.Forward(x)
	x = Add(x, residual)

	// Feed-forward with residual connection
	residual = x
	x = block.LN2.Forward(x)
	x = block.FFN.Forward(x)
	x = Add(x, residual)

	return x
}

// FeedForward implements the feed-forward network
type FeedForward struct {
	W1        *Tensor // [hidden, ffn_dim] or [hidden, 2*ffn_dim] for SwiGLU
	B1        *Tensor // [ffn_dim] or [2*ffn_dim] for SwiGLU
	W2        *Tensor // [ffn_dim, hidden]
	B2        *Tensor // [hidden]
	Hidden    int
	FFNDim    int
	UseSwiGLU bool // If true, use SwiGLU instead of GELU
}

// Forward applies the feed-forward network
func (ffn *FeedForward) Forward(x *Tensor) *Tensor {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]

	// Flatten for matrix multiplication
	xFlat := x.Reshape(batchSize*seqLen, ffn.Hidden)

	// First linear layer
	x = MatMul(xFlat, ffn.W1)

	if ffn.UseSwiGLU {
		// SwiGLU: W1 projects to 2*ffn_dim, split into value and gate
		// x shape: [batch*seq, 2*ffn_dim]
		halfDim := ffn.FFNDim

		// Split into value and gate
		value := x.SliceLastDim(0, halfDim)
		gate := x.SliceLastDim(halfDim, 2*halfDim)

		// Apply SiLU to gate and multiply with value
		gate = SiLU(gate)
		x = NewTensor(value.Shape...)
		for i := 0; i < len(value.Data); i++ {
			x.Data[i] = value.Data[i] * gate.Data[i]
		}
	} else {
		// Standard GELU activation
		// Add bias
		if ffn.B1 != nil {
			for i := 0; i < batchSize*seqLen; i++ {
				for j := 0; j < ffn.FFNDim; j++ {
					x.Data[i*ffn.FFNDim+j] += ffn.B1.Data[j]
				}
			}
		}
		x = GELU(x)
	}

	// Second linear layer
	x = MatMul(x, ffn.W2)

	// Add bias
	if ffn.B2 != nil {
		for i := 0; i < batchSize*seqLen; i++ {
			for j := 0; j < ffn.Hidden; j++ {
				x.Data[i*ffn.Hidden+j] += ffn.B2.Data[j]
			}
		}
	}

	// Reshape back
	x = x.Reshape(batchSize, seqLen, ffn.Hidden)

	return x
}

// LayerNormLayer wraps layer normalization with parameters
type LayerNormLayer struct {
	Weight *Tensor
	Bias   *Tensor
	Eps    float32
}

// Forward applies layer normalization
func (ln *LayerNormLayer) Forward(x *Tensor) *Tensor {
	return LayerNorm(x, ln.Weight, ln.Bias, ln.Eps)
}
