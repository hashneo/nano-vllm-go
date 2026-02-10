package tensor

// TransformerModel is a generic transformer that adapts to different architectures
type TransformerModel struct {
	Config *ModelConfig

	// Embeddings
	TokenEmbedding *Tensor // [vocab_size, hidden]
	PosEmbedding   *Tensor // [max_seq_len, hidden] - only if PositionLearned

	// Transformer blocks
	Blocks []*GenericBlock

	// Final layer norm
	LNFinal *LayerNormLayer

	// LM head
	LMHead *Tensor // [hidden, vocab_size] (or tied with TokenEmbedding)
}

// GenericBlock adapts to different block styles (sequential or parallel)
type GenericBlock struct {
	Config    *ModelConfig
	Attention AttentionLayer // Can be MHA, MQA, or GQA
	FFN       *FeedForward
	InputLN   *LayerNormLayer // Input norm (used by parallel blocks)
	AttnLN    *LayerNormLayer // Attention norm (used by sequential blocks)
	FFNLN     *LayerNormLayer // FFN norm (used by sequential blocks)
}

// AttentionLayer interface for different attention types
type AttentionLayer interface {
	Forward(x *Tensor) *Tensor
	SetConfig(config *ModelConfig)
}

// NewTransformerModel creates a generic transformer model
func NewTransformerModel(config *ModelConfig) *TransformerModel {
	model := &TransformerModel{
		Config: config,
		Blocks: make([]*GenericBlock, config.NumLayers),
	}

	// Create transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		// For hybrid architectures (Granite), check layer type
		if len(config.HybridLayers) > 0 && i < len(config.HybridLayers) {
			model.Blocks[i] = NewGenericBlockWithType(config, config.HybridLayers[i])
		} else {
			model.Blocks[i] = NewGenericBlock(config)
		}
	}

	// Create final norm
	model.LNFinal = &LayerNormLayer{Eps: config.NormEps}

	return model
}

// NewGenericBlock creates a block that adapts to the config
func NewGenericBlock(config *ModelConfig) *GenericBlock {
	return NewGenericBlockWithType(config, "attention")
}

// NewGenericBlockWithType creates a block of specified type (attention or mamba2)
func NewGenericBlockWithType(config *ModelConfig, layerType string) *GenericBlock {
	block := &GenericBlock{
		Config: config,
	}

	// Create layer based on type
	// Accept both "mamba" (Granite convention) and "mamba2" (our internal name)
	if layerType == "mamba2" || layerType == "mamba" {
		// Create Mamba2 layer
		block.Attention = NewMamba2Layer(config)

		// Mamba2 layers in Granite also have FFN (shared_mlp) and norms
		block.FFN = &FeedForward{
			Hidden:    config.Hidden,
			FFNDim:    config.FFNDim,
			UseSwiGLU: config.ActivationType == ActivationSwiGLU,
		}
		// Use AttnLN and FFNLN for the two norms
		block.AttnLN = &LayerNormLayer{Eps: config.NormEps}
		block.FFNLN = &LayerNormLayer{Eps: config.NormEps}
	} else {
		// Create attention based on type
		switch config.AttentionType {
		case AttentionMQA:
			block.Attention = NewMultiQueryAttention(config.NumHeads, config.HeadDim, config.MaxSeqLen)
		case AttentionMHA:
			block.Attention = &MultiHeadAttention{
				NumHeads: config.NumHeads,
				HeadDim:  config.HeadDim,
				Hidden:   config.Hidden,
			}
		case AttentionGQA:
			// Grouped-Query Attention: multiple KV heads
			block.Attention = &GroupedQueryAttention{
				NumHeads:   config.NumHeads,
				NumKVHeads: config.NumKVHeads,
				HeadDim:    config.HeadDim,
				Hidden:     config.Hidden,
			}
		}

		// Create FFN (only for attention layers, Mamba2 has its own)
		block.FFN = &FeedForward{
			Hidden:    config.Hidden,
			FFNDim:    config.FFNDim,
			UseSwiGLU: config.ActivationType == ActivationSwiGLU,
		}

		// Create norms based on block style
		if config.BlockStyle == BlockParallel {
			// Parallel: single input norm
			block.InputLN = &LayerNormLayer{Eps: config.NormEps}
		} else {
			// Sequential: separate norms for attention and FFN
			block.AttnLN = &LayerNormLayer{Eps: config.NormEps}
			block.FFNLN = &LayerNormLayer{Eps: config.NormEps}
		}
	}

	return block
}

// Forward applies the transformer block based on block style
func (block *GenericBlock) Forward(x *Tensor) *Tensor {
	// Debug: Uncomment to trace layer execution
	// if _, isMamba := block.Attention.(*Mamba2Layer); isMamba {
	// 	fmt.Println("DEBUG: Executing Mamba2 layer")
	// } else {
	// 	fmt.Println("DEBUG: Executing Attention layer")
	// }

	// Check if this is a Mamba2 layer by checking the Attention type
	if _, isMamba := block.Attention.(*Mamba2Layer); isMamba {
		// Mamba2 layer - has special forward pass
		return block.forwardMamba2(x)
	}

	if block.Config.BlockStyle == BlockParallel {
		return block.forwardParallel(x)
	}
	return block.forwardSequential(x)
}

// forwardMamba2: Mamba2 layer with residual connection
func (block *GenericBlock) forwardMamba2(x *Tensor) *Tensor {
	// Granite's Mamba2 layers follow sequential style like attention layers:
	// 1. Norm + Mamba2 + residual
	// 2. Norm + FFN + residual

	// Mamba2 with residual
	residual := x
	if block.AttnLN != nil {
		x = block.AttnLN.Forward(x)
	}
	x = block.Attention.Forward(x)
	x = Add(x, residual)

	// FFN with residual (Granite has shared_mlp for Mamba2 layers)
	if block.FFN != nil {
		residual = x
		if block.FFNLN != nil {
			x = block.FFNLN.Forward(x)
		}
		x = block.FFN.Forward(x)
		x = Add(x, residual)
	}

	return x
}

// forwardSequential: attention then FFN (GPT-2, Llama style)
func (block *GenericBlock) forwardSequential(x *Tensor) *Tensor {
	// Self-attention with residual
	residual := x
	x = block.AttnLN.Forward(x)
	x = block.Attention.Forward(x)
	x = Add(x, residual)

	// Feed-forward with residual
	residual = x
	x = block.FFNLN.Forward(x)
	x = block.FFN.Forward(x)
	x = Add(x, residual)

	return x
}

// forwardParallel: attention and FFN in parallel (Falcon style)
func (block *GenericBlock) forwardParallel(x *Tensor) *Tensor {
	residual := x

	// Single layer norm at input
	x = block.InputLN.Forward(x)

	// Run attention and FFN in parallel (conceptually)
	attnOut := block.Attention.Forward(x)
	ffnOut := block.FFN.Forward(x)

	// Add both outputs to residual
	for i := range residual.Data {
		residual.Data[i] = residual.Data[i] + attnOut.Data[i] + ffnOut.Data[i]
	}

	return residual
}

// Forward performs a forward pass through the model
func (m *TransformerModel) Forward(tokenIDs []int) *Tensor {
	logits, _ := m.ForwardWithCache(tokenIDs, nil, 0)
	return logits
}

// ForwardWithCache performs a forward pass with KV caching support
// posOffset is the position offset for positional embeddings when using cache
// Returns: (logits, updated_kv_cache)
func (m *TransformerModel) ForwardWithCache(tokenIDs []int, kvCache *KVCache, posOffset int) (*Tensor, *KVCache) {
	batchSize := 1
	seqLen := len(tokenIDs)

	// Create KV cache if not provided
	if kvCache == nil {
		kvCache = NewKVCache(m.Config.NumLayers)
	}

	// 1. Embeddings (with position offset for cached generation)
	x := m.embedWithOffset(tokenIDs, posOffset)

	// 2. Apply transformer blocks with KV caching
	for i, block := range m.Blocks {
		// Check if this block supports KV caching (has attention)
		if mha, ok := block.Attention.(*MultiHeadAttention); ok {
			// Get cached K,V for this layer
			kCache, vCache := kvCache.GetLayer(i)

			// Forward with cache
			var newK, newV *Tensor
			residual := x
			if block.AttnLN != nil {
				x = block.AttnLN.Forward(x)
			}
			x, newK, newV = mha.ForwardWithCache(x, kCache, vCache)
			x = Add(x, residual)

			// Update cache
			kvCache.SetLayer(i, newK, newV)

			// FFN
			if block.FFN != nil {
				residual = x
				if block.FFNLN != nil {
					x = block.FFNLN.Forward(x)
				}
				x = block.FFN.Forward(x)
				x = Add(x, residual)
			}
		} else {
			// Non-attention blocks (e.g., Mamba2) don't use KV cache
			x = block.Forward(x)
		}
	}

	// 3. Final layer norm
	x = m.LNFinal.Forward(x)

	// 4. LM head projection
	xFlat := x.Reshape(batchSize*seqLen, m.Config.Hidden)
	var logits *Tensor
	if m.Config.TiedEmbedding {
		// Use transposed token embedding
		logits = MatMul(xFlat, Transpose(m.TokenEmbedding))
	} else {
		logits = MatMul(xFlat, m.LMHead)
	}
	logits = logits.Reshape(batchSize, seqLen, m.Config.VocabSize)

	// Apply logits scaling if set (Granite muP scaling)
	if m.Config.LogitsScaling != 0 {
		for i := range logits.Data {
			logits.Data[i] *= m.Config.LogitsScaling
		}
	}

	return logits, kvCache
}

// embed creates embeddings based on position type
func (m *TransformerModel) embed(tokenIDs []int) *Tensor {
	return m.embedWithOffset(tokenIDs, 0)
}

// embedWithOffset creates embeddings with position offset for KV caching
func (m *TransformerModel) embedWithOffset(tokenIDs []int, posOffset int) *Tensor {
	seqLen := len(tokenIDs)
	hidden := m.Config.Hidden

	result := NewTensor(1, seqLen, hidden)

	for i, tokenID := range tokenIDs {
		// Token embedding
		for j := 0; j < hidden; j++ {
			embVal := m.TokenEmbedding.Data[tokenID*hidden+j]
			// Apply embedding multiplier if set (Granite muP scaling)
			if m.Config.EmbeddingMultiplier != 0 {
				embVal *= m.Config.EmbeddingMultiplier
			}
			result.Data[i*hidden+j] = embVal
		}

		// Add position embedding if using learned positions
		if m.Config.PositionType == PositionLearned && m.PosEmbedding != nil {
			actualPos := posOffset + i
			if actualPos < m.Config.MaxSeqLen {
				for j := 0; j < hidden; j++ {
					result.Data[i*hidden+j] += m.PosEmbedding.Data[actualPos*hidden+j]
				}
			}
		}
		// Note: RoPE is applied in attention layer, not here
	}

	return result
}

// GetLogitsForLastToken returns logits for the last token
func (m *TransformerModel) GetLogitsForLastToken(logits *Tensor) []float32 {
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	lastTokenLogits := make([]float32, vocabSize)
	offset := (seqLen - 1) * vocabSize
	copy(lastTokenLogits, logits.Data[offset:offset+vocabSize])

	return lastTokenLogits
}

// PrintInfo prints model information
func (m *TransformerModel) PrintInfo() {
	m.Config.PrintInfo()
}

// SetConfig allows attention layers to access config
func (mha *MultiHeadAttention) SetConfig(config *ModelConfig) {
	// MHA doesn't need config (no RoPE)
}

// SetConfig allows MQA to access config for RoPE
func (mqa *MultiQueryAttention) SetConfig(config *ModelConfig) {
	// Already has RoPE cache
}
