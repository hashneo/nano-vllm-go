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
		// SwiGLU: W1 is concatenated [gateWeight | upWeight]
		// After MatMul(x, W1), we get [gate | up]
		// x shape: [batch*seq, 2*ffn_dim]
		halfDim := ffn.FFNDim

		// Split: first half is gate (apply SiLU), second half is up (no activation)
		gate := x.SliceLastDim(0, halfDim)
		up := x.SliceLastDim(halfDim, 2*halfDim)

		// Apply SiLU to gate and multiply with up
		// SwiGLU: silu(gate) * up
		gate = SiLU(gate)
		x = NewTensor(gate.Shape...)
		for i := 0; i < len(gate.Data); i++ {
			x.Data[i] = gate.Data[i] * up.Data[i]
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
