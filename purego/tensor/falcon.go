package tensor

// FalconConfig holds Falcon model configuration
type FalconConfig struct {
	VocabSize  int
	Hidden     int
	NumLayers  int
	NumHeads   int
	FFNDim     int
	MaxSeqLen  int
	EOSTokenID int
}

// FalconBlock implements Falcon's parallel attention + FFN architecture
// Key difference from GPT-2: Attention and FFN run in parallel, not sequential!
type FalconBlock struct {
	Attention *MultiQueryAttention
	FFN       *FeedForward
	InputLN   *LayerNormLayer // Single layer norm before parallel blocks
}

// Forward applies the Falcon block with parallel architecture
func (block *FalconBlock) Forward(x *Tensor) *Tensor {
	// Falcon uses parallel attention + FFN (not sequential!)
	residual := x

	// Single layer norm at input
	x = block.InputLN.Forward(x)

	// Run attention and FFN in parallel (conceptually)
	// In practice, we run sequentially but add both to residual
	attnOut := block.Attention.Forward(x)
	ffnOut := block.FFN.Forward(x)

	// Add both outputs to residual
	// output = residual + attn(ln(x)) + ffn(ln(x))
	for i := range residual.Data {
		residual.Data[i] = residual.Data[i] + attnOut.Data[i] + ffnOut.Data[i]
	}

	return residual
}

// FalconModel implements the Falcon architecture
type FalconModel struct {
	Config *FalconConfig

	// Embeddings (no position embeddings - using RoPE instead!)
	TokenEmbedding *Tensor // [vocab_size, hidden]

	// Transformer blocks
	Blocks []*FalconBlock

	// Final layer norm
	LNFinal *LayerNormLayer

	// LM head
	LMHead *Tensor // [hidden, vocab_size]
}

// NewFalconModel creates a new Falcon model
func NewFalconModel(config *FalconConfig) *FalconModel {
	model := &FalconModel{
		Config: config,
		Blocks: make([]*FalconBlock, config.NumLayers),
	}

	headDim := config.Hidden / config.NumHeads

	// Create transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		model.Blocks[i] = &FalconBlock{
			Attention: NewMultiQueryAttention(config.NumHeads, headDim, config.MaxSeqLen),
			FFN: &FeedForward{
				Hidden: config.Hidden,
				FFNDim: config.FFNDim,
			},
			InputLN: &LayerNormLayer{Eps: 1e-5},
		}
	}

	model.LNFinal = &LayerNormLayer{Eps: 1e-5}

	return model
}

// Forward performs a forward pass through Falcon
func (m *FalconModel) Forward(tokenIDs []int) *Tensor {
	batchSize := 1
	seqLen := len(tokenIDs)

	// 1. Token embeddings only (no position embeddings - RoPE handles position)
	x := m.embedTokens(tokenIDs)

	// 2. Apply Falcon blocks (parallel attention + FFN)
	for _, block := range m.Blocks {
		x = block.Forward(x)
	}

	// 3. Final layer norm
	x = m.LNFinal.Forward(x)

	// 4. LM head projection
	xFlat := x.Reshape(batchSize*seqLen, m.Config.Hidden)
	logits := MatMul(xFlat, m.LMHead)
	logits = logits.Reshape(batchSize, seqLen, m.Config.VocabSize)

	return logits
}

// embedTokens creates token embeddings (no position embeddings!)
func (m *FalconModel) embedTokens(tokenIDs []int) *Tensor {
	seqLen := len(tokenIDs)
	hidden := m.Config.Hidden

	result := NewTensor(1, seqLen, hidden)

	for i, tokenID := range tokenIDs {
		// Only token embedding (RoPE provides positional information)
		for j := 0; j < hidden; j++ {
			result.Data[i*hidden+j] = m.TokenEmbedding.Data[tokenID*hidden+j]
		}
	}

	return result
}

// GetLogitsForLastToken returns logits for the last token
func (m *FalconModel) GetLogitsForLastToken(logits *Tensor) []float32 {
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	lastTokenLogits := make([]float32, vocabSize)
	offset := (seqLen - 1) * vocabSize
	copy(lastTokenLogits, logits.Data[offset:offset+vocabSize])

	return lastTokenLogits
}

// PrintInfo prints model information
func (m *FalconModel) PrintInfo() {
	fmt := func(format string, args ...interface{}) {
		// Using standard fmt would cause import cycle
		println(format)
	}

	fmt("Falcon Model Configuration:")
	fmt("  Vocabulary: %d", m.Config.VocabSize)
	fmt("  Hidden size: %d", m.Config.Hidden)
	fmt("  Layers: %d", m.Config.NumLayers)
	fmt("  Heads: %d (Multi-Query Attention)", m.Config.NumHeads)
	fmt("  FFN dim: %d", m.Config.FFNDim)
	fmt("  Max seq len: %d", m.Config.MaxSeqLen)

	// Count parameters
	params := 0
	params += m.Config.VocabSize * m.Config.Hidden // token embedding
	// No position embedding (RoPE is computed on the fly)

	// Each layer:
	// - Q projection: hidden * hidden
	// - KV projection: hidden * (head_dim * 2) where head_dim = hidden/num_heads
	// - Out projection: hidden * hidden
	// - FFN: 2 * hidden * ffn_dim
	headDim := m.Config.Hidden / m.Config.NumHeads
	paramsPerLayer := 0
	paramsPerLayer += m.Config.Hidden * m.Config.Hidden     // Q
	paramsPerLayer += m.Config.Hidden * (headDim * 2)       // KV (shared!)
	paramsPerLayer += m.Config.Hidden * m.Config.Hidden     // Out
	paramsPerLayer += 2 * m.Config.Hidden * m.Config.FFNDim // FFN
	params += m.Config.NumLayers * paramsPerLayer

	params += m.Config.Hidden * m.Config.VocabSize // LM head

	println("  Parameters: ~", params/1000000, "M")
}
