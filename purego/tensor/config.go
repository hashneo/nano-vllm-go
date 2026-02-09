package tensor

// ModelArchitecture defines the neural network architecture
type ModelArchitecture string

const (
	ArchGPT2    ModelArchitecture = "gpt2"     // GPT-2: MHA, learned positions, sequential blocks
	ArchFalcon  ModelArchitecture = "falcon"   // Falcon: MQA, RoPE, parallel blocks
	ArchLlama   ModelArchitecture = "llama"    // Llama: GQA, RoPE, RMSNorm, SwiGLU
	ArchMistral ModelArchitecture = "mistral"  // Mistral: GQA, RoPE, RMSNorm, SwiGLU, sliding window
	ArchGranite ModelArchitecture = "granite"  // Granite: Hybrid attention + Mamba2 layers
)

// AttentionType defines the attention mechanism
type AttentionType string

const (
	AttentionMHA AttentionType = "mha" // Multi-Head: separate K,V per head
	AttentionMQA AttentionType = "mqa" // Multi-Query: shared K,V across all heads
	AttentionGQA AttentionType = "gqa" // Grouped-Query: shared K,V per group
)

// NormType defines the normalization layer
type NormType string

const (
	NormLayer NormType = "layernorm" // LayerNorm: mean + variance
	NormRMS   NormType = "rmsnorm"   // RMSNorm: RMS only (simpler)
)

// PositionType defines position encoding
type PositionType string

const (
	PositionLearned PositionType = "learned" // Learned embeddings (GPT-2)
	PositionRoPE    PositionType = "rope"    // Rotary (Falcon, Llama, Mistral)
	PositionALiBi   PositionType = "alibi"   // Attention with Linear Biases
	PositionNoPE    PositionType = "nope"    // No positional encoding (Granite/Mamba2)
)

// ActivationType defines activation function
type ActivationType string

const (
	ActivationGELU   ActivationType = "gelu"   // GELU (GPT-2, Falcon)
	ActivationSwiGLU ActivationType = "swiglu" // SwiGLU (Llama, Mistral)
)

// BlockStyle defines how attention and FFN are combined
type BlockStyle string

const (
	BlockSequential BlockStyle = "sequential" // Attn then FFN (GPT-2, Llama)
	BlockParallel   BlockStyle = "parallel"   // Attn and FFN in parallel (Falcon)
)

// ModelConfig holds universal model configuration
type ModelConfig struct {
	// Model identity
	Architecture ModelArchitecture
	ModelName    string

	// Model dimensions
	VocabSize  int
	Hidden     int
	NumLayers  int
	NumHeads   int   // Number of query heads
	NumKVHeads int   // Number of KV heads (1 for MQA, same as NumHeads for MHA)
	HeadDim    int   // Usually hidden / num_heads
	FFNDim     int   // Usually 4 * hidden
	MaxSeqLen  int

	// Architecture choices
	AttentionType  AttentionType
	NormType       NormType
	PositionType   PositionType
	ActivationType ActivationType
	BlockStyle     BlockStyle

	// Special tokens
	EOSTokenID int
	BOSTokenID int
	PadTokenID int

	// RoPE parameters
	RoPEBase      float64 // Usually 10000.0
	RoPEScaling   float64 // For extended context (usually 1.0)

	// Normalization parameters
	NormEps       float32 // Usually 1e-5 or 1e-6

	// Other
	TiedEmbedding bool // Whether LM head shares weights with token embedding

	// Mamba2 parameters (for hybrid architectures like Granite)
	Mamba2Expand      int     // Expansion factor (usually 2)
	Mamba2StateSize   int     // State dimension (usually 128)
	Mamba2NumHeads    int     // Number of SSM heads (usually 48)
	Mamba2NGroups     int     // Number of groups (usually 8)
	Mamba2ConvKernel  int     // Causal conv kernel size (usually 4)
	Mamba2DtRank      int     // Delta rank (auto: ceil(hidden/16))
	Mamba2ChunkSize   int     // Chunk size for scanning (usually 256)
	Mamba2TimeStepMin float32 // Min time step (usually 0.001)
	Mamba2TimeStepMax float32 // Max time step (usually 0.1)

	// Hybrid architecture (for Granite)
	HybridLayers      []string // Layer types: "attention" or "mamba2"
	NumAttentionLayers int     // Number of attention layers
	NumMamba2Layers    int     // Number of Mamba2 layers
}

// NewGPT2Config creates config for GPT-2 models
func NewGPT2Config() *ModelConfig {
	return &ModelConfig{
		Architecture:   ArchGPT2,
		ModelName:      "gpt2",
		VocabSize:      50257,
		Hidden:         768,
		NumLayers:      12,
		NumHeads:       12,
		NumKVHeads:     12, // MHA: same as query heads
		HeadDim:        64,
		FFNDim:         3072,
		MaxSeqLen:      1024,
		AttentionType:  AttentionMHA,
		NormType:       NormLayer,
		PositionType:   PositionLearned,
		ActivationType: ActivationGELU,
		BlockStyle:     BlockSequential,
		EOSTokenID:     50256,
		BOSTokenID:     50256,
		PadTokenID:     50256,
		NormEps:        1e-5,
		TiedEmbedding:  true,
	}
}

// NewFalconConfig creates config for Falcon models
func NewFalconConfig(size string) *ModelConfig {
	config := &ModelConfig{
		Architecture:   ArchFalcon,
		AttentionType:  AttentionMQA,
		NormType:       NormLayer,
		PositionType:   PositionRoPE,
		ActivationType: ActivationGELU,
		BlockStyle:     BlockParallel,
		NumKVHeads:     1, // MQA: single KV head
		RoPEBase:       10000.0,
		NormEps:        1e-5,
		TiedEmbedding:  false,
		EOSTokenID:     11,
		BOSTokenID:     11,
		PadTokenID:     11,
	}

	switch size {
	case "7b":
		config.ModelName = "falcon-7b"
		config.VocabSize = 65024
		config.Hidden = 4544
		config.NumLayers = 32
		config.NumHeads = 71
		config.HeadDim = 64
		config.FFNDim = 18176
		config.MaxSeqLen = 2048
	case "40b":
		config.ModelName = "falcon-40b"
		config.VocabSize = 65024
		config.Hidden = 8192
		config.NumLayers = 60
		config.NumHeads = 128
		config.HeadDim = 64
		config.FFNDim = 32768
		config.MaxSeqLen = 2048
	default:
		// Default to 7b
		return NewFalconConfig("7b")
	}

	return config
}

// NewLlamaConfig creates config for Llama models
func NewLlamaConfig(size string) *ModelConfig {
	config := &ModelConfig{
		Architecture:   ArchLlama,
		AttentionType:  AttentionGQA,
		NormType:       NormRMS,
		PositionType:   PositionRoPE,
		ActivationType: ActivationSwiGLU,
		BlockStyle:     BlockSequential,
		RoPEBase:       10000.0,
		NormEps:        1e-6,
		TiedEmbedding:  false,
		EOSTokenID:     2,
		BOSTokenID:     1,
		PadTokenID:     0,
	}

	switch size {
	case "7b":
		config.ModelName = "llama-7b"
		config.VocabSize = 32000
		config.Hidden = 4096
		config.NumLayers = 32
		config.NumHeads = 32
		config.NumKVHeads = 8 // GQA: 8 KV heads for 32 query heads
		config.HeadDim = 128
		config.FFNDim = 11008
		config.MaxSeqLen = 4096
	case "13b":
		config.ModelName = "llama-13b"
		config.VocabSize = 32000
		config.Hidden = 5120
		config.NumLayers = 40
		config.NumHeads = 40
		config.NumKVHeads = 10
		config.HeadDim = 128
		config.FFNDim = 13824
		config.MaxSeqLen = 4096
	default:
		return NewLlamaConfig("7b")
	}

	return config
}

// NewGraniteConfig creates config for IBM Granite 4 models
// Granite uses hybrid architecture with 4 attention layers + many Mamba2 layers
func NewGraniteConfig(size string) *ModelConfig {
	config := &ModelConfig{
		Architecture:   ArchGranite,
		AttentionType:  AttentionGQA,
		NormType:       NormRMS,
		PositionType:   PositionNoPE, // Granite uses no positional encoding
		ActivationType: ActivationSwiGLU,
		BlockStyle:     BlockSequential,
		RoPEBase:       10000.0,
		NormEps:        1e-5,
		TiedEmbedding:  true,

		// Mamba2 parameters
		Mamba2Expand:      2,
		Mamba2StateSize:   128,
		Mamba2ConvKernel:  4,
		Mamba2ChunkSize:   256,
		Mamba2TimeStepMin: 0.001,
		Mamba2TimeStepMax: 0.1,
	}

	switch size {
	case "350m":
		config.ModelName = "granite-4.0-h-350m"
		config.VocabSize = 49152
		config.Hidden = 768
		config.NumLayers = 32
		config.NumHeads = 12
		config.NumKVHeads = 4 // GQA: 4 KV heads
		config.HeadDim = 64
		config.FFNDim = 2048
		config.MaxSeqLen = 32768

		// Mamba2 specific
		config.Mamba2NumHeads = 48
		config.Mamba2NGroups = 8
		config.Mamba2DtRank = 0 // Auto: ceil(768/16) = 48

		// Hybrid: 4 attention + 28 Mamba2 (interspersed)
		config.NumAttentionLayers = 4
		config.NumMamba2Layers = 28
		config.HybridLayers = make([]string, 32)
		// Attention at layers 10, 13, 17, 27; rest are Mamba2
		attentionLayers := []int{10, 13, 17, 27}
		for i := 0; i < 32; i++ {
			config.HybridLayers[i] = "mamba2"
		}
		for _, idx := range attentionLayers {
			config.HybridLayers[idx] = "attention"
		}

	case "1b":
		config.ModelName = "granite-4.0-h-1b"
		config.VocabSize = 49152
		config.Hidden = 1536
		config.NumLayers = 40
		config.NumHeads = 12
		config.NumKVHeads = 4 // GQA: 4 KV heads
		config.HeadDim = 128
		config.FFNDim = 4096
		config.MaxSeqLen = 128000

		// Mamba2 specific
		config.Mamba2NumHeads = 48
		config.Mamba2NGroups = 8
		config.Mamba2DtRank = 0 // Auto: ceil(1536/16) = 96

		// Hybrid: 4 attention + 36 Mamba2
		config.NumAttentionLayers = 4
		config.NumMamba2Layers = 36
		config.HybridLayers = make([]string, 40)
		// First 4 are attention, rest are Mamba2
		for i := 0; i < 4; i++ {
			config.HybridLayers[i] = "attention"
		}
		for i := 4; i < 40; i++ {
			config.HybridLayers[i] = "mamba2"
		}

	default:
		return NewGraniteConfig("350m")
	}

	return config
}

// PrintInfo prints model configuration
func (c *ModelConfig) PrintInfo() {
	println("Model Configuration:")
	println("  Name:", c.ModelName)
	println("  Architecture:", string(c.Architecture))
	println("  Vocabulary:", c.VocabSize)
	println("  Hidden size:", c.Hidden)
	println("  Layers:", c.NumLayers)
	println("  Query heads:", c.NumHeads)
	println("  KV heads:", c.NumKVHeads)
	println("  Head dimension:", c.HeadDim)
	println("  FFN dimension:", c.FFNDim)
	println("  Max sequence:", c.MaxSeqLen)
	println("  Attention:", string(c.AttentionType))
	println("  Position:", string(c.PositionType))
	println("  Normalization:", string(c.NormType))
	println("  Activation:", string(c.ActivationType))
	println("  Block style:", string(c.BlockStyle))

	// Calculate parameters
	params := c.EstimateParameters()
	if params > 1e9 {
		println("  Parameters: ~", params/1e9, "B")
	} else {
		println("  Parameters: ~", params/1e6, "M")
	}
}

// EstimateParameters estimates total parameter count
func (c *ModelConfig) EstimateParameters() int64 {
	params := int64(0)

	// Token embedding
	params += int64(c.VocabSize * c.Hidden)

	// Position embedding (if learned)
	if c.PositionType == PositionLearned {
		params += int64(c.MaxSeqLen * c.Hidden)
	}

	// Transformer layers
	paramsPerLayer := int64(0)

	// Attention weights
	switch c.AttentionType {
	case AttentionMHA:
		// Q, K, V, Out all full size
		paramsPerLayer += 4 * int64(c.Hidden*c.Hidden)
	case AttentionMQA:
		// Q is full size, K,V are small
		paramsPerLayer += int64(c.Hidden * c.Hidden)                    // Q
		paramsPerLayer += 2 * int64(c.Hidden*c.HeadDim)                 // K, V
		paramsPerLayer += int64(c.Hidden * c.Hidden)                    // Out
	case AttentionGQA:
		// Q is full, K,V are medium
		paramsPerLayer += int64(c.Hidden * c.Hidden)                    // Q
		paramsPerLayer += 2 * int64(c.Hidden*c.NumKVHeads*c.HeadDim)   // K, V
		paramsPerLayer += int64(c.Hidden * c.Hidden)                    // Out
	}

	// FFN weights
	if c.ActivationType == ActivationSwiGLU {
		// SwiGLU has 3 weight matrices
		paramsPerLayer += 3 * int64(c.Hidden*c.FFNDim)
	} else {
		// Standard FFN has 2 weight matrices
		paramsPerLayer += 2 * int64(c.Hidden*c.FFNDim)
	}

	// Normalization weights (weight + bias, or just weight for RMS)
	if c.NormType == NormRMS {
		paramsPerLayer += 2 * int64(c.Hidden) // 2 norms per layer, weight only
	} else {
		paramsPerLayer += 4 * int64(c.Hidden) // 2 norms per layer, weight + bias
	}

	params += int64(c.NumLayers) * paramsPerLayer

	// Final norm
	if c.NormType == NormRMS {
		params += int64(c.Hidden)
	} else {
		params += 2 * int64(c.Hidden)
	}

	// LM head (unless tied with embedding)
	if !c.TiedEmbedding {
		params += int64(c.Hidden * c.VocabSize)
	}

	return params
}
