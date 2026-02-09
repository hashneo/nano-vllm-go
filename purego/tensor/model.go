package tensor

import "fmt"

// GPT2Config holds model configuration
type GPT2Config struct {
	VocabSize  int
	Hidden     int
	NumLayers  int
	NumHeads   int
	FFNDim     int
	MaxSeqLen  int
	EOSTokenID int
}

// GPT2Model implements a GPT-2 style transformer
type GPT2Model struct {
	Config *GPT2Config

	// Embeddings
	TokenEmbedding *Tensor // [vocab_size, hidden]
	PosEmbedding   *Tensor // [max_seq_len, hidden]

	// Transformer blocks
	Blocks []*TransformerBlock

	// Final layer norm
	LNFinal *LayerNormLayer

	// LM head (tied with token embedding in GPT-2)
	LMHead *Tensor // [hidden, vocab_size]
}

// NewGPT2Model creates a new GPT-2 model
func NewGPT2Model(config *GPT2Config) *GPT2Model {
	model := &GPT2Model{
		Config: config,
		Blocks: make([]*TransformerBlock, config.NumLayers),
	}

	headDim := config.Hidden / config.NumHeads

	// Create transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		model.Blocks[i] = &TransformerBlock{
			Attention: &MultiHeadAttention{
				NumHeads: config.NumHeads,
				HeadDim:  headDim,
				Hidden:   config.Hidden,
			},
			FFN: &FeedForward{
				Hidden: config.Hidden,
				FFNDim: config.FFNDim,
			},
			LN1: &LayerNormLayer{Eps: 1e-5},
			LN2: &LayerNormLayer{Eps: 1e-5},
		}
	}

	return model
}

// Forward performs a forward pass through the model
func (m *GPT2Model) Forward(tokenIDs []int) *Tensor {
	batchSize := 1
	seqLen := len(tokenIDs)

	// 1. Token + position embeddings
	x := m.embed(tokenIDs)

	// 2. Apply transformer blocks
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

// embed creates embeddings for tokens
func (m *GPT2Model) embed(tokenIDs []int) *Tensor {
	seqLen := len(tokenIDs)
	hidden := m.Config.Hidden

	result := NewTensor(1, seqLen, hidden)

	for i, tokenID := range tokenIDs {
		// Token embedding
		for j := 0; j < hidden; j++ {
			tokEmbed := m.TokenEmbedding.Data[tokenID*hidden+j]
			posEmbed := m.PosEmbedding.Data[i*hidden+j]
			result.Data[i*hidden+j] = tokEmbed + posEmbed
		}
	}

	return result
}

// GetLogitsForLastToken returns logits for the last token
func (m *GPT2Model) GetLogitsForLastToken(logits *Tensor) []float32 {
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	// Extract last position
	lastTokenLogits := make([]float32, vocabSize)
	offset := (seqLen - 1) * vocabSize
	copy(lastTokenLogits, logits.Data[offset:offset+vocabSize])

	return lastTokenLogits
}

// PrintInfo prints model information
func (m *GPT2Model) PrintInfo() {
	fmt.Printf("GPT-2 Model Configuration:\n")
	fmt.Printf("  Vocabulary: %d\n", m.Config.VocabSize)
	fmt.Printf("  Hidden size: %d\n", m.Config.Hidden)
	fmt.Printf("  Layers: %d\n", m.Config.NumLayers)
	fmt.Printf("  Heads: %d\n", m.Config.NumHeads)
	fmt.Printf("  FFN dim: %d\n", m.Config.FFNDim)
	fmt.Printf("  Max seq len: %d\n", m.Config.MaxSeqLen)

	// Count parameters
	params := 0
	params += m.Config.VocabSize * m.Config.Hidden                        // token embedding
	params += m.Config.MaxSeqLen * m.Config.Hidden                        // position embedding
	params += m.Config.NumLayers * 12 * m.Config.Hidden * m.Config.Hidden // attention (Q,K,V,O x 3 weights + biases)
	params += m.Config.NumLayers * 2 * m.Config.Hidden * m.Config.FFNDim  // FFN
	params += m.Config.Hidden * m.Config.VocabSize                        // LM head

	fmt.Printf("  Parameters: ~%.1fM\n", float64(params)/1e6)
}
