package tensor

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

// WeightMapping defines how to map safetensors keys to model components
type WeightMapping struct {
	// Embedding keys
	TokenEmbeddingKey string // e.g., "wte.weight", "transformer.word_embeddings.weight"
	PosEmbeddingKey   string // e.g., "wpe.weight" (empty for RoPE models)
	LMHeadKey         string // e.g., "lm_head.weight" (empty if tied)

	// Layer key templates (use {layer} as placeholder)
	LayerPrefix     string // e.g., "h.{layer}", "transformer.h.{layer}"
	AttentionQKey   string // e.g., ".attn.c_attn.weight", ".self_attention.query_key_value.weight"
	AttentionKVKey  string // For MQA/GQA with separate KV (empty if combined with Q)
	AttentionOutKey string // e.g., ".attn.c_proj.weight"
	FFNUpKey        string // e.g., ".mlp.c_fc.weight", ".mlp.dense_h_to_4h.weight"
	FFNDownKey      string // e.g., ".mlp.c_proj.weight", ".mlp.dense_4h_to_h.weight"
	AttnNormKey     string // e.g., ".ln_1.weight"
	FFNNormKey      string // e.g., ".ln_2.weight"
	InputNormKey    string // For parallel blocks, e.g., ".input_layernorm.weight"
	PostAttnNormKey string // Post-attention norm (Granite)

	// Final norm keys
	FinalNormKey string // e.g., "ln_f.weight", "transformer.ln_f.weight"

	// Combined QKV handling
	QKVCombined bool // Whether Q,K,V are in one weight matrix
	QKVSplitFn  func(*Tensor, *ModelConfig) (*Tensor, *Tensor, *Tensor)

	// Weight format
	TransposeWeights bool // If true, weights are in PyTorch format [out, in] and need transposing

	// Mamba2 keys (for hybrid architectures like Granite)
	Mamba2ALogKey      string // e.g., ".mamba.A_log"
	Mamba2DKey         string // e.g., ".mamba.D"
	Mamba2Conv1dWeight string // e.g., ".mamba.conv1d.weight"
	Mamba2Conv1dBias   string // e.g., ".mamba.conv1d.bias"
	Mamba2DtBiasKey    string // e.g., ".mamba.dt_bias"
	Mamba2InProjKey    string // e.g., ".mamba.in_proj.weight"
	Mamba2NormKey      string // e.g., ".mamba.norm.weight"
	Mamba2OutProjKey   string // e.g., ".mamba.out_proj.weight"
}

// GetGPT2Mapping returns weight mapping for GPT-2
func GetGPT2Mapping() *WeightMapping {
	return &WeightMapping{
		TokenEmbeddingKey: "wte.weight",
		PosEmbeddingKey:   "wpe.weight",
		LMHeadKey:         "", // Tied with token embedding
		LayerPrefix:       "h.{layer}",
		AttentionQKey:     ".attn.c_attn.weight",
		AttentionOutKey:   ".attn.c_proj.weight",
		FFNUpKey:          ".mlp.c_fc.weight",
		FFNDownKey:        ".mlp.c_proj.weight",
		AttnNormKey:       ".ln_1.weight",
		FFNNormKey:        ".ln_2.weight",
		FinalNormKey:      "ln_f.weight",
		QKVCombined:       true,
		QKVSplitFn:        splitGPT2QKV,
		// Note: GPT-2 safetensors already has weights transposed to [in, out]
	}
}

// GetFalconMapping returns weight mapping for Falcon
func GetFalconMapping() *WeightMapping {
	return &WeightMapping{
		TokenEmbeddingKey: "transformer.word_embeddings.weight",
		PosEmbeddingKey:   "", // Uses RoPE
		LMHeadKey:         "lm_head.weight",
		LayerPrefix:       "transformer.h.{layer}",
		AttentionQKey:     ".self_attention.query_key_value.weight",
		AttentionOutKey:   ".self_attention.dense.weight",
		FFNUpKey:          ".mlp.dense_h_to_4h.weight",
		FFNDownKey:        ".mlp.dense_4h_to_h.weight",
		InputNormKey:      ".input_layernorm.weight", // Parallel block
		FinalNormKey:      "transformer.ln_f.weight",
		QKVCombined:       true,
		QKVSplitFn:        splitFalconQKV,
		TransposeWeights:  true, // Falcon uses PyTorch format [out, in], needs transpose
	}
}

// GetLlamaMapping returns weight mapping for Llama
func GetLlamaMapping() *WeightMapping {
	return &WeightMapping{
		TokenEmbeddingKey: "model.embed_tokens.weight",
		PosEmbeddingKey:   "", // Uses RoPE
		LMHeadKey:         "lm_head.weight",
		LayerPrefix:       "model.layers.{layer}",
		AttentionQKey:     ".self_attn.q_proj.weight",
		AttentionKVKey:    ".self_attn.k_proj.weight", // Separate K,V for GQA
		AttentionOutKey:   ".self_attn.o_proj.weight",
		FFNUpKey:          ".mlp.gate_proj.weight", // SwiGLU gate
		FFNDownKey:        ".mlp.down_proj.weight",
		AttnNormKey:       ".input_layernorm.weight",
		FFNNormKey:        ".post_attention_layernorm.weight",
		FinalNormKey:      "model.norm.weight",
		QKVCombined:       false, // Llama has separate Q, K, V
		TransposeWeights:  true,  // Llama uses PyTorch format [out, in], needs transposing to [in, out]
	}
}

// GetGraniteMapping returns weight mapping for Granite (hybrid Mamba2 + Attention)
func GetGraniteMapping() *WeightMapping {
	return &WeightMapping{
		TokenEmbeddingKey: "model.embed_tokens.weight",
		PosEmbeddingKey:   "", // Uses NoPE (no positional encoding)
		LMHeadKey:         "", // Tied with token embedding
		LayerPrefix:       "model.layers.{layer}",

		// Attention keys (for attention layers)
		AttentionQKey:   ".self_attn.q_proj.weight",
		AttentionKVKey:  ".self_attn.k_proj.weight", // Separate K,V for GQA
		AttentionOutKey: ".self_attn.o_proj.weight",

		// Shared MLP (used by both attention and Mamba2 layers)
		FFNUpKey:   ".shared_mlp.input_linear.weight",
		FFNDownKey: ".shared_mlp.output_linear.weight",

		// Norms
		InputNormKey:    ".input_layernorm.weight",
		PostAttnNormKey: ".post_attention_layernorm.weight",
		FinalNormKey:    "model.norm.weight",

		// Mamba2 keys (for Mamba2 layers)
		Mamba2ALogKey:      ".mamba.A_log",
		Mamba2DKey:         ".mamba.D",
		Mamba2Conv1dWeight: ".mamba.conv1d.weight",
		Mamba2Conv1dBias:   ".mamba.conv1d.bias",
		Mamba2DtBiasKey:    ".mamba.dt_bias",
		Mamba2InProjKey:    ".mamba.in_proj.weight",
		Mamba2NormKey:      ".mamba.norm.weight",
		Mamba2OutProjKey:   ".mamba.out_proj.weight",

		QKVCombined:      false, // Granite has separate Q, K, V
		TransposeWeights: true,  // Granite uses PyTorch format, needs transpose
	}
}

// LoadModel loads a transformer model from safetensors
func LoadModel(modelPath string, config *ModelConfig) (*TransformerModel, error) {
	// Read safetensors file
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	// Parse header
	headerSize := binary.LittleEndian.Uint64(data[:8])
	headerBytes := data[8 : 8+headerSize]
	tensorData := data[8+headerSize:]

	var metadata map[string]TensorInfo
	if err := json.Unmarshal(headerBytes, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse metadata: %w", err)
	}

	fmt.Printf("Loading model with %d tensors...\n", len(metadata))

	// Get weight mapping for this architecture
	var mapping *WeightMapping
	switch config.Architecture {
	case ArchGPT2:
		mapping = GetGPT2Mapping()
	case ArchFalcon:
		mapping = GetFalconMapping()
	case ArchLlama:
		mapping = GetLlamaMapping()
	case ArchGranite:
		mapping = GetGraniteMapping()
	default:
		return nil, fmt.Errorf("unsupported architecture: %s", config.Architecture)
	}

	// Create model
	model := NewTransformerModel(config)

	// Load embeddings
	if err := loadTensorRequired(tensorData, metadata, mapping.TokenEmbeddingKey, &model.TokenEmbedding); err != nil {
		return nil, fmt.Errorf("failed to load token embedding: %w", err)
	}

	if mapping.PosEmbeddingKey != "" {
		loadTensorOptional(tensorData, metadata, mapping.PosEmbeddingKey, &model.PosEmbedding)
	}

	// Load transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		if err := loadBlock(tensorData, metadata, mapping, model.Blocks[i], i, config); err != nil {
			return nil, fmt.Errorf("failed to load layer %d: %w", i, err)
		}
	}

	// Load final norm
	if err := loadNorm(tensorData, metadata, mapping.FinalNormKey, model.LNFinal); err != nil {
		return nil, fmt.Errorf("failed to load final norm: %w", err)
	}

	// Load LM head
	if mapping.LMHeadKey != "" && !config.TiedEmbedding {
		loadTensorOptional(tensorData, metadata, mapping.LMHeadKey, &model.LMHead)
		// Transpose if PyTorch format
		if model.LMHead != nil && mapping.TransposeWeights {
			model.LMHead = Transpose(model.LMHead)
		}
	}

	// If tied, use transposed token embedding
	if config.TiedEmbedding {
		model.LMHead = Transpose(model.TokenEmbedding)
	}

	fmt.Printf("✓ Model loaded successfully\n")
	model.PrintInfo()

	return model, nil
}

// loadBlock loads weights for a single transformer block
func loadBlock(tensorData []byte, metadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, layer int, config *ModelConfig) error {
	// Replace {layer} with actual layer number
	prefix := strings.ReplaceAll(mapping.LayerPrefix, "{layer}", fmt.Sprintf("%d", layer))

	// Check if this is a hybrid architecture (Granite)
	isMamba2 := false
	if len(config.HybridLayers) > layer {
		// Granite uses "mamba" in config, but we call them "mamba2" internally
		layerType := config.HybridLayers[layer]
		isMamba2 = layerType == "mamba2" || layerType == "mamba"
	}

	if isMamba2 {
		// Load Mamba2 layer
		if err := loadMamba2(tensorData, metadata, mapping, block, prefix, config); err != nil {
			return fmt.Errorf("mamba2: %w", err)
		}

		// Mamba2 layers still have shared MLP
		if err := loadFFN(tensorData, metadata, mapping, block, prefix); err != nil {
			return fmt.Errorf("FFN: %w", err)
		}

		// Load norms for Mamba2 layer
		inputNormKey := prefix + mapping.InputNormKey
		if inputNormKey == "" || mapping.InputNormKey == "" {
			return fmt.Errorf("input norm key is empty (prefix=%s, InputNormKey=%s)", prefix, mapping.InputNormKey)
		}
		if err := loadNorm(tensorData, metadata, inputNormKey, block.AttnLN); err != nil {
			return fmt.Errorf("input norm (key=%s): %w", inputNormKey, err)
		}
		if mapping.PostAttnNormKey != "" {
			postNormKey := prefix + mapping.PostAttnNormKey
			if err := loadNorm(tensorData, metadata, postNormKey, block.FFNLN); err != nil {
				return fmt.Errorf("post attn norm (key=%s): %w", postNormKey, err)
			}
		}
	} else {
		// Load attention weights
		if err := loadAttention(tensorData, metadata, mapping, block, prefix, config); err != nil {
			return fmt.Errorf("attention: %w", err)
		}

		// Load FFN weights
		if err := loadFFN(tensorData, metadata, mapping, block, prefix); err != nil {
			return fmt.Errorf("FFN: %w", err)
		}

		// Load norms
		if config.BlockStyle == BlockParallel {
			if err := loadNorm(tensorData, metadata, prefix+mapping.InputNormKey, block.InputLN); err != nil {
				return fmt.Errorf("input norm: %w", err)
			}
		} else {
			// Sequential: load two norms
			// For Granite, use InputNormKey if AttnNormKey is empty
			attnNormKey := mapping.AttnNormKey
			if attnNormKey == "" {
				attnNormKey = mapping.InputNormKey
			}
			if err := loadNorm(tensorData, metadata, prefix+attnNormKey, block.AttnLN); err != nil {
				return fmt.Errorf("attn norm: %w", err)
			}

			// For second norm, prefer PostAttnNormKey, then FFNNormKey
			if mapping.PostAttnNormKey != "" {
				if err := loadNorm(tensorData, metadata, prefix+mapping.PostAttnNormKey, block.FFNLN); err != nil {
					return fmt.Errorf("FFN norm: %w", err)
				}
			} else if err := loadNorm(tensorData, metadata, prefix+mapping.FFNNormKey, block.FFNLN); err != nil {
				return fmt.Errorf("FFN norm: %w", err)
			}
		}
	}

	return nil
}

// loadAttention loads attention weights
func loadAttention(tensorData []byte, metadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string, config *ModelConfig) error {
	switch attn := block.Attention.(type) {
	case *MultiQueryAttention:
		// MQA: Load combined QKV or separate
		if mapping.QKVCombined {
			var qkvWeight *Tensor
			if err := loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionQKey, &qkvWeight); err != nil {
				return err
			}
			// Transpose BEFORE splitting if PyTorch format
			if mapping.TransposeWeights {
				qkvWeight = Transpose(qkvWeight)
			}
			// Split into Q and KV
			if mapping.QKVSplitFn != nil {
				Q, K, V := mapping.QKVSplitFn(qkvWeight, config)
				attn.QWeight = Q
				attn.KVWeight = combineMQAKV(K, V)
			}
		} else {
			loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionQKey, &attn.QWeight)
			loadTensorOptional(tensorData, metadata, prefix+mapping.AttentionKVKey, &attn.KVWeight)
			// Transpose if PyTorch format
			if mapping.TransposeWeights {
				attn.QWeight = Transpose(attn.QWeight)
				if attn.KVWeight != nil {
					attn.KVWeight = Transpose(attn.KVWeight)
				}
			}
		}
		loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)

		// Transpose output weight if PyTorch format
		if mapping.TransposeWeights {
			attn.OutWeight = Transpose(attn.OutWeight)
		}

	case *GroupedQueryAttention:
		// GQA: Load separate Q, K, V weights
		loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionQKey, &attn.QWeight)
		loadTensorRequired(tensorData, metadata, prefix+".self_attn.k_proj.weight", &attn.KWeight)
		loadTensorRequired(tensorData, metadata, prefix+".self_attn.v_proj.weight", &attn.VWeight)
		loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)

		// Transpose if PyTorch format
		if mapping.TransposeWeights {
			attn.QWeight = Transpose(attn.QWeight)
			attn.KWeight = Transpose(attn.KWeight)
			attn.VWeight = Transpose(attn.VWeight)
			attn.OutWeight = Transpose(attn.OutWeight)
		}

	case *MultiHeadAttention:
		// MHA: Load combined or separate Q,K,V
		if mapping.QKVCombined {
			var qkvWeight *Tensor
			if err := loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionQKey, &qkvWeight); err != nil {
				return err
			}
			if mapping.QKVSplitFn != nil {
				Q, K, V := mapping.QKVSplitFn(qkvWeight, config)
				attn.QWeight = Q
				attn.KWeight = K
				attn.VWeight = V
			}
			// Load and split combined QKV bias
			// Remove ".weight" suffix and add ".bias"
			qBiasKey := strings.Replace(mapping.AttentionQKey, ".weight", ".bias", 1)
			var qkvBias *Tensor
			loadTensorOptional(tensorData, metadata, prefix+qBiasKey, &qkvBias)
			if qkvBias != nil {
				hidden := config.Hidden
				attn.QBias = &Tensor{Data: qkvBias.Data[0:hidden], Shape: []int{hidden}}
				attn.KBias = &Tensor{Data: qkvBias.Data[hidden : 2*hidden], Shape: []int{hidden}}
				attn.VBias = &Tensor{Data: qkvBias.Data[2*hidden : 3*hidden], Shape: []int{hidden}}
			}
		} else {
			loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionQKey, &attn.QWeight)
			loadTensorOptional(tensorData, metadata, prefix+".self_attn.k_proj.weight", &attn.KWeight)
			loadTensorOptional(tensorData, metadata, prefix+".self_attn.v_proj.weight", &attn.VWeight)
			// Load separate biases
			loadTensorOptional(tensorData, metadata, prefix+mapping.AttentionQKey+".bias", &attn.QBias)
			loadTensorOptional(tensorData, metadata, prefix+".self_attn.k_proj.bias", &attn.KBias)
			loadTensorOptional(tensorData, metadata, prefix+".self_attn.v_proj.bias", &attn.VBias)
		}
		loadTensorRequired(tensorData, metadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)
		// Remove ".weight" suffix and add ".bias"
		outBiasKey := strings.Replace(mapping.AttentionOutKey, ".weight", ".bias", 1)
		loadTensorOptional(tensorData, metadata, prefix+outBiasKey, &attn.OutBias)


		// Transpose if PyTorch format
		if mapping.TransposeWeights {
			attn.QWeight = Transpose(attn.QWeight)
			if attn.KWeight != nil {
				attn.KWeight = Transpose(attn.KWeight)
			}
			if attn.VWeight != nil {
				attn.VWeight = Transpose(attn.VWeight)
			}
			attn.OutWeight = Transpose(attn.OutWeight)
		}
	}

	return nil
}

// loadMamba2 loads Mamba2 layer weights
func loadMamba2(tensorData []byte, metadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string, config *ModelConfig) error {
	mamba, ok := block.Attention.(*Mamba2Layer)
	if !ok {
		return fmt.Errorf("block does not contain Mamba2Layer")
	}

	// Load SSM parameters
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2ALogKey, &mamba.ALog); err != nil {
		return fmt.Errorf("A_log: %w", err)
	}
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2DKey, &mamba.D); err != nil {
		return fmt.Errorf("D: %w", err)
	}

	// Load conv weights
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2Conv1dWeight, &mamba.ConvWeight); err != nil {
		return fmt.Errorf("conv weight: %w", err)
	}
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2Conv1dBias, &mamba.ConvBias); err != nil {
		return fmt.Errorf("conv bias: %w", err)
	}

	// Load delta bias
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2DtBiasKey, &mamba.DeltaBias); err != nil {
		return fmt.Errorf("dt bias: %w", err)
	}

	// Load projection weights
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2InProjKey, &mamba.InProj); err != nil {
		return fmt.Errorf("in_proj: %w", err)
	}
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2NormKey, &mamba.Norm); err != nil {
		return fmt.Errorf("norm: %w", err)
	}
	if err := loadTensorRequired(tensorData, metadata, prefix+mapping.Mamba2OutProjKey, &mamba.OutProj); err != nil {
		return fmt.Errorf("out_proj: %w", err)
	}

	// NOTE: Testing without transpose - weights might already be in correct format
	// if mapping.TransposeWeights {
	// 	mamba.InProj = Transpose(mamba.InProj)
	// 	mamba.OutProj = Transpose(mamba.OutProj)
	// }

	// Note: Granite's in_proj contains everything (x, z, delta, B, C)
	// XProj and DtProj are not separate tensors in Granite
	// The forward pass extracts them from InProj

	return nil
}

// loadFFN loads FFN weights
func loadFFN(tensorData []byte, metadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string) error {
	if block.FFN == nil {
		return nil // Skip if no FFN (shouldn't happen for Granite)
	}

	// For SwiGLU with separate gate/up projections (Llama/Mistral)
	if block.FFN.UseSwiGLU {
		// Try loading separate gate and up projections
		var gateWeight, upWeight *Tensor
		gateKey := prefix + mapping.FFNUpKey                    // gate_proj
		upKey := strings.Replace(gateKey, "gate_proj", "up_proj", 1) // up_proj

		errGate := loadTensorFromData(tensorData, metadata, gateKey, &gateWeight)
		errUp := loadTensorFromData(tensorData, metadata, upKey, &upWeight)

		if errGate == nil && errUp == nil {
			// Both found - concatenate them
			// Transpose first if PyTorch format
			if mapping.TransposeWeights {
				gateWeight = Transpose(gateWeight)
				upWeight = Transpose(upWeight)
			}
			// Concatenate along last dimension: [hidden, ffn_dim] + [hidden, ffn_dim] -> [hidden, 2*ffn_dim]
			block.FFN.W1 = ConcatenateLastDim(gateWeight, upWeight)
		} else {
			// Fall back to single weight (combined gate+up)
			loadTensorRequired(tensorData, metadata, prefix+mapping.FFNUpKey, &block.FFN.W1)
			if mapping.TransposeWeights {
				block.FFN.W1 = Transpose(block.FFN.W1)
			}
		}
	} else {
		// Standard FFN
		loadTensorRequired(tensorData, metadata, prefix+mapping.FFNUpKey, &block.FFN.W1)
		if mapping.TransposeWeights {
			block.FFN.W1 = Transpose(block.FFN.W1)
		}
	}

	loadTensorRequired(tensorData, metadata, prefix+mapping.FFNDownKey, &block.FFN.W2)
	// Transpose if PyTorch format [out, in] -> [in, out]
	if mapping.TransposeWeights {
		block.FFN.W2 = Transpose(block.FFN.W2)
	}

	// Biases (optional)
	loadTensorOptional(tensorData, metadata, prefix+mapping.FFNUpKey+".bias", &block.FFN.B1)
	loadTensorOptional(tensorData, metadata, prefix+mapping.FFNDownKey+".bias", &block.FFN.B2)

	return nil
}

// loadNorm loads normalization layer weights
func loadNorm(tensorData []byte, metadata map[string]TensorInfo, key string, norm *LayerNormLayer) error {
	if err := loadTensorRequired(tensorData, metadata, key, &norm.Weight); err != nil {
		return err
	}
	// Bias is optional (RMSNorm doesn't have bias)
	// Remove ".weight" suffix and add ".bias"
	biasKey := strings.Replace(key, ".weight", ".bias", 1)
	loadTensorOptional(tensorData, metadata, biasKey, &norm.Bias)
	return nil
}

// loadTensorRequired loads a tensor and returns error if not found
func loadTensorRequired(data []byte, metadata map[string]TensorInfo, name string, target **Tensor) error {
	if err := loadTensorFromData(data, metadata, name, target); err != nil {
		return fmt.Errorf("required tensor '%s' not found: %w", name, err)
	}
	return nil
}

// loadTensorOptional loads a tensor, ignores if not found
func loadTensorOptional(data []byte, metadata map[string]TensorInfo, name string, target **Tensor) {
	loadTensorFromData(data, metadata, name, target)
}

// loadTensorFromData loads a single tensor from safetensors data
func loadTensorFromData(data []byte, metadata map[string]TensorInfo, name string, target **Tensor) error {
	info, ok := metadata[name]
	if !ok {
		// Try with "transformer." prefix (for newer HF models)
		altName := "transformer." + name
		info, ok = metadata[altName]
		if !ok {
			return fmt.Errorf("tensor not found: %s (also tried: %s)", name, altName)
		}
	}

	// Extract tensor data
	start := info.Offset[0]
	end := info.Offset[1]
	tensorBytes := data[start:end]

	// Calculate number of elements
	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	tensorData := make([]float32, numElements)

	// Convert based on dtype
	switch info.Dtype {
	case "F32":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(tensorBytes[i*4 : (i+1)*4])
			tensorData[i] = float32FromBits(bits)
		}
	case "F16":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(tensorBytes[i*2 : (i+1)*2])
			tensorData[i] = float32FromFloat16(bits)
		}
	case "BF16":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(tensorBytes[i*2 : (i+1)*2])
			tensorData[i] = float32FromBFloat16(bits)
		}
	default:
		return fmt.Errorf("unsupported dtype: %s", info.Dtype)
	}

	*target = &Tensor{
		Data:  tensorData,
		Shape: info.Shape,
	}

	return nil
}

// splitGPT2QKV splits GPT-2's combined QKV weight
func splitGPT2QKV(qkvWeight *Tensor, config *ModelConfig) (*Tensor, *Tensor, *Tensor) {
	// GPT-2 format: [hidden, 3*hidden]
	// Q/K/V are split along columns (dim 1), not rows
	// Column layout: [Q_cols: 0-767, K_cols: 768-1535, V_cols: 1536-2303]
	hidden := config.Hidden

	Q := NewTensor(hidden, hidden)
	K := NewTensor(hidden, hidden)
	V := NewTensor(hidden, hidden)

	// Extract column-wise splits
	for row := 0; row < hidden; row++ {
		srcRowOffset := row * (3 * hidden)
		// Copy Q columns (0:768)
		for col := 0; col < hidden; col++ {
			Q.Data[row*hidden+col] = qkvWeight.Data[srcRowOffset+col]
		}
		// Copy K columns (768:1536)
		for col := 0; col < hidden; col++ {
			K.Data[row*hidden+col] = qkvWeight.Data[srcRowOffset+hidden+col]
		}
		// Copy V columns (1536:2304)
		for col := 0; col < hidden; col++ {
			V.Data[row*hidden+col] = qkvWeight.Data[srcRowOffset+2*hidden+col]
		}
	}

	return Q, K, V
}

// splitFalconQKV splits Falcon's combined QKV weight for MQA
func splitFalconQKV(qkvWeight *Tensor, config *ModelConfig) (*Tensor, *Tensor, *Tensor) {
	// Falcon format: [hidden, hidden + 2*head_dim]
	// Q is full size, K,V are single head
	hidden := config.Hidden
	headDim := config.HeadDim

	Q := &Tensor{
		Data:  qkvWeight.Data[0 : hidden*hidden],
		Shape: []int{hidden, hidden},
	}
	K := &Tensor{
		Data:  qkvWeight.Data[hidden*hidden : hidden*(hidden+headDim)],
		Shape: []int{hidden, headDim},
	}
	V := &Tensor{
		Data:  qkvWeight.Data[hidden*(hidden+headDim) : hidden*(hidden+2*headDim)],
		Shape: []int{hidden, headDim},
	}

	return Q, K, V
}

// combineMQAKV combines separate K and V into MQA KV weight
func combineMQAKV(K, V *Tensor) *Tensor {
	// Combine K,V into single tensor [hidden, head_dim*2]
	hidden := K.Shape[0]
	headDim := K.Shape[1]

	combined := NewTensor(hidden, headDim*2)
	for i := 0; i < hidden; i++ {
		for j := 0; j < headDim; j++ {
			combined.Data[i*headDim*2+j] = K.Data[i*headDim+j]
			combined.Data[i*headDim*2+headDim+j] = V.Data[i*headDim+j]
		}
	}

	return combined
}

// Helper functions for type conversion

func float32FromBits(bits uint32) float32 {
	ptr := &bits
	return *(*float32)(unsafe.Pointer(ptr))
}

func float32FromFloat16(bits uint16) float32 {
	sign := uint32((bits >> 15) & 1)
	exp := uint32((bits >> 10) & 0x1F)
	frac := uint32(bits & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			return float32FromBits(sign << 31)
		}
		// Subnormal
		exp = 127 - 14
		for (frac & 0x400) == 0 {
			frac <<= 1
			exp--
		}
		frac &= 0x3FF
	} else if exp == 0x1F {
		// Inf or NaN
		exp = 0xFF
	} else {
		// Normal
		exp += 127 - 15
	}

	result := (sign << 31) | (exp << 23) | (frac << 13)
	return float32FromBits(result)
}

func float32FromBFloat16(bits uint16) float32 {
	// BF16 is just truncated FP32 (sign + 8-bit exp + 7-bit mantissa)
	return float32FromBits(uint32(bits) << 16)
}

// LoadModelConfig loads model config from JSON file
func LoadModelConfig(configPath string) (*ModelConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	// Try to infer architecture from JSON
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Create config based on architecture field or infer from structure
	arch, _ := raw["architecture"].(string)

	var config *ModelConfig
	switch arch {
	case "gpt2":
		config = NewGPT2Config()
	case "falcon":
		config = NewFalconConfig("7b")
	case "llama":
		config = NewLlamaConfig("7b")
	default:
		// Try to infer from fields
		config = inferConfigFromJSON(raw)
	}

	// Override with JSON values
	if v, ok := raw["vocab_size"].(float64); ok {
		config.VocabSize = int(v)
	}
	// GPT-2 uses "n_embd" instead of "hidden_size"
	if v, ok := raw["n_embd"].(float64); ok {
		config.Hidden = int(v)
	}
	if v, ok := raw["hidden_size"].(float64); ok {
		config.Hidden = int(v)
	}
	// GPT-2 uses "n_layer" instead of "num_hidden_layers"
	if v, ok := raw["n_layer"].(float64); ok {
		config.NumLayers = int(v)
	}
	if v, ok := raw["num_hidden_layers"].(float64); ok {
		config.NumLayers = int(v)
	}
	if v, ok := raw["num_layers"].(float64); ok {
		config.NumLayers = int(v)
	}
	// GPT-2 uses "n_head" instead of "num_attention_heads"
	if v, ok := raw["n_head"].(float64); ok {
		config.NumHeads = int(v)
	}
	if v, ok := raw["num_attention_heads"].(float64); ok {
		config.NumHeads = int(v)
	}
	if v, ok := raw["num_heads"].(float64); ok {
		config.NumHeads = int(v)
	}
	if v, ok := raw["num_key_value_heads"].(float64); ok {
		config.NumKVHeads = int(v)
	}
	if v, ok := raw["num_kv_heads"].(float64); ok {
		config.NumKVHeads = int(v)
	}
	if v, ok := raw["head_dim"].(float64); ok {
		config.HeadDim = int(v)
	}
	// If HeadDim not specified, calculate from hidden / num_heads
	if config.HeadDim == 0 && config.Hidden > 0 && config.NumHeads > 0 {
		config.HeadDim = config.Hidden / config.NumHeads
	}
	if v, ok := raw["eos_token_id"].(float64); ok {
		config.EOSTokenID = int(v)
	}
	if v, ok := raw["bos_token_id"].(float64); ok {
		config.BOSTokenID = int(v)
	}
	if v, ok := raw["pad_token_id"].(float64); ok {
		config.PadTokenID = int(v)
	}
	if v, ok := raw["rope_theta"].(float64); ok {
		config.RoPEBase = v
	}
	if v, ok := raw["rms_norm_eps"].(float64); ok {
		config.NormEps = float32(v)
	}
	if v, ok := raw["layer_norm_epsilon"].(float64); ok {
		config.NormEps = float32(v)
	}
	// GPT-2 uses "n_inner" for FFN dimension (can be null, defaults to 4*hidden)
	if v, ok := raw["n_inner"].(float64); ok {
		config.FFNDim = int(v)
	}
	if v, ok := raw["intermediate_size"].(float64); ok {
		config.FFNDim = int(v)
	}
	// If FFNDim is still 0, default to 4 * hidden_size (GPT-2 convention)
	if config.FFNDim == 0 && config.Hidden > 0 {
		config.FFNDim = 4 * config.Hidden
	}
	if v, ok := raw["tie_word_embeddings"].(bool); ok {
		config.TiedEmbedding = v
	}

	// Granite-specific fields
	if layerTypes, ok := raw["layer_types"].([]interface{}); ok {
		config.HybridLayers = make([]string, len(layerTypes))
		for i, lt := range layerTypes {
			if layerType, ok := lt.(string); ok {
				config.HybridLayers[i] = layerType
			}
		}
	}
	if v, ok := raw["mamba_expand"].(float64); ok {
		config.Mamba2Expand = int(v)
	}
	if v, ok := raw["mamba_d_state"].(float64); ok {
		config.Mamba2StateSize = int(v)
	}
	if v, ok := raw["mamba_n_heads"].(float64); ok {
		config.Mamba2NumHeads = int(v)
	}
	if v, ok := raw["mamba_d_head"].(float64); ok {
		config.Mamba2HeadDim = int(v)
	}
	if v, ok := raw["mamba_n_groups"].(float64); ok {
		config.Mamba2NGroups = int(v)
	}
	if v, ok := raw["mamba_d_conv"].(float64); ok {
		config.Mamba2ConvKernel = int(v)
	}
	if v, ok := raw["mamba_chunk_size"].(float64); ok {
		config.Mamba2ChunkSize = int(v)
	}

	// muP scaling parameters (for Granite)
	if v, ok := raw["embedding_multiplier"].(float64); ok {
		config.EmbeddingMultiplier = float32(v)
	}
	if v, ok := raw["attention_multiplier"].(float64); ok {
		config.AttentionMultiplier = float32(v)
	}
	if v, ok := raw["residual_multiplier"].(float64); ok {
		config.ResidualMultiplier = float32(v)
	}
	if v, ok := raw["logits_scaling"].(float64); ok {
		config.LogitsScaling = float32(v)
	}

	return config, nil
}

// inferConfigFromJSON tries to infer architecture from JSON structure
func inferConfigFromJSON(raw map[string]interface{}) *ModelConfig {
	// Check for model_type field
	if modelType, ok := raw["model_type"].(string); ok {
		switch modelType {
		case "gpt2":
			return NewGPT2Config()
		case "falcon", "RefinedWeb", "RefinedWebModel":
			return NewFalconConfig("7b")
		case "llama", "LlamaForCausalLM":
			return NewLlamaConfig("7b")
		case "granitemoehybrid":
			// Detect Granite size from hidden_size
			if hidden, ok := raw["hidden_size"].(float64); ok {
				if hidden <= 800 {
					return NewGraniteConfig("350m")
				}
				return NewGraniteConfig("1b")
			}
			return NewGraniteConfig("350m")
		}
	}

	// Default to GPT-2 style
	return NewGPT2Config()
}

// ShardedModelIndex represents the index file for sharded models
type ShardedModelIndex struct {
	Metadata  map[string]interface{} `json:"metadata"`
	WeightMap map[string]string      `json:"weight_map"`
}

// LoadModelFromDirectory loads model from a directory with config.json and model.safetensors
func LoadModelFromDirectory(dir string) (*TransformerModel, error) {
	// Try to load config
	configPath := filepath.Join(dir, "config.json")
	config, err := LoadModelConfig(configPath)
	if err != nil {
		// Try model_info.json (our custom format)
		configPath = filepath.Join(dir, "model_info.json")
		config, err = LoadModelConfig(configPath)
		if err != nil {
			return nil, fmt.Errorf("no valid config found: %w", err)
		}
	}

	// Check if this is a sharded model (has model.safetensors.index.json)
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		// Load sharded model
		return LoadShardedModel(dir, indexPath, config)
	}

	// Load single model file
	modelPath := filepath.Join(dir, "model.safetensors")
	return LoadModel(modelPath, config)
}

// LoadShardedModel loads a model from multiple safetensors shards
func LoadShardedModel(modelDir string, indexPath string, config *ModelConfig) (*TransformerModel, error) {
	// Read index file
	indexData, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read index file: %w", err)
	}

	var index ShardedModelIndex
	if err := json.Unmarshal(indexData, &index); err != nil {
		return nil, fmt.Errorf("failed to parse index file: %w", err)
	}

	fmt.Printf("Loading sharded model with %d tensors across multiple files...\n", len(index.WeightMap))

	// Group tensors by shard file
	shardTensors := make(map[string][]string)
	for tensorName, shardFile := range index.WeightMap {
		shardTensors[shardFile] = append(shardTensors[shardFile], tensorName)
	}

	fmt.Printf("Found %d shard files to load\n", len(shardTensors))

	// Load all shards and build combined metadata
	allMetadata := make(map[string]TensorInfo)
	allTensorData := make(map[string][]byte)

	for shardFile := range shardTensors {
		shardPath := filepath.Join(modelDir, shardFile)
		fmt.Printf("Loading shard: %s...\n", shardFile)

		data, err := os.ReadFile(shardPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read shard %s: %w", shardFile, err)
		}

		// Parse shard header
		headerSize := binary.LittleEndian.Uint64(data[:8])
		headerBytes := data[8 : 8+headerSize]
		shardTensorData := data[8+headerSize:]

		var shardMetadata map[string]TensorInfo
		if err := json.Unmarshal(headerBytes, &shardMetadata); err != nil {
			return nil, fmt.Errorf("failed to parse shard metadata: %w", err)
		}

		// Add tensors from this shard to combined metadata
		for tensorName, tensorInfo := range shardMetadata {
			if tensorName == "__metadata__" {
				continue
			}
			// Store tensor metadata (offsets are relative to this shard's data section)
			allMetadata[tensorName] = tensorInfo
			// Store reference to this shard's data
			allTensorData[tensorName] = shardTensorData
		}
	}

	fmt.Printf("Loaded %d tensors from shards\n", len(allMetadata))

	// Get weight mapping for this architecture
	var mapping *WeightMapping
	switch config.Architecture {
	case ArchGPT2:
		mapping = GetGPT2Mapping()
	case ArchFalcon:
		mapping = GetFalconMapping()
	case ArchLlama:
		mapping = GetLlamaMapping()
	case ArchGranite:
		mapping = GetGraniteMapping()
	default:
		return nil, fmt.Errorf("unsupported architecture: %s", config.Architecture)
	}

	// Create model
	model := NewTransformerModel(config)

	// Load embeddings
	if err := loadTensorFromShards(allTensorData, allMetadata, mapping.TokenEmbeddingKey, &model.TokenEmbedding); err != nil {
		return nil, fmt.Errorf("failed to load token embedding: %w", err)
	}

	if mapping.PosEmbeddingKey != "" {
		loadTensorFromShardsOptional(allTensorData, allMetadata, mapping.PosEmbeddingKey, &model.PosEmbedding)
	}

	// Load transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		if err := loadBlockFromShards(allTensorData, allMetadata, mapping, model.Blocks[i], i, config); err != nil {
			return nil, fmt.Errorf("failed to load layer %d: %w", i, err)
		}
	}

	// Load final norm
	if err := loadNormFromShards(allTensorData, allMetadata, mapping.FinalNormKey, model.LNFinal); err != nil {
		return nil, fmt.Errorf("failed to load final norm: %w", err)
	}

	// Load LM head
	if mapping.LMHeadKey != "" && !config.TiedEmbedding {
		loadTensorFromShardsOptional(allTensorData, allMetadata, mapping.LMHeadKey, &model.LMHead)
		// Transpose if PyTorch format
		if model.LMHead != nil && mapping.TransposeWeights {
			model.LMHead = Transpose(model.LMHead)
		}
	}

	// If tied, use transposed token embedding
	if config.TiedEmbedding {
		model.LMHead = Transpose(model.TokenEmbedding)
	}

	fmt.Printf("✓ Model loaded successfully\n")
	model.PrintInfo()

	return model, nil
}

// loadTensorFromShards loads a tensor from sharded data
func loadTensorFromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, name string, target **Tensor) error {
	info, ok := allMetadata[name]
	if !ok {
		// Try with "transformer." prefix
		altName := "transformer." + name
		info, ok = allMetadata[altName]
		if !ok {
			return fmt.Errorf("tensor not found: %s (also tried: %s)", name, altName)
		}
		name = altName
	}

	tensorBytes := allData[name]

	// Extract tensor data
	start := info.Offset[0]
	end := info.Offset[1]
	tensorBytesSlice := tensorBytes[start:end]

	// Calculate number of elements
	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	tensorData := make([]float32, numElements)

	// Convert based on dtype
	switch info.Dtype {
	case "F32":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(tensorBytesSlice[i*4 : (i+1)*4])
			tensorData[i] = float32FromBits(bits)
		}
	case "F16":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(tensorBytesSlice[i*2 : (i+1)*2])
			tensorData[i] = float32FromFloat16(bits)
		}
	case "BF16":
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(tensorBytesSlice[i*2 : (i+1)*2])
			tensorData[i] = float32FromBFloat16(bits)
		}
	default:
		return fmt.Errorf("unsupported dtype: %s", info.Dtype)
	}

	*target = &Tensor{
		Data:  tensorData,
		Shape: info.Shape,
	}

	return nil
}

// loadTensorFromShardsOptional loads a tensor, ignores if not found
func loadTensorFromShardsOptional(allData map[string][]byte, allMetadata map[string]TensorInfo, name string, target **Tensor) {
	loadTensorFromShards(allData, allMetadata, name, target)
}

// loadBlockFromShards loads weights for a single transformer block from shards
func loadBlockFromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, layer int, config *ModelConfig) error {
	// Replace {layer} with actual layer number
	prefix := strings.ReplaceAll(mapping.LayerPrefix, "{layer}", fmt.Sprintf("%d", layer))

	// Check if this is a hybrid architecture (Granite)
	isMamba2 := false
	if len(config.HybridLayers) > layer {
		layerType := config.HybridLayers[layer]
		isMamba2 = layerType == "mamba2" || layerType == "mamba"
	}

	if isMamba2 {
		// Load Mamba2 layer
		if err := loadMamba2FromShards(allData, allMetadata, mapping, block, prefix, config); err != nil {
			return fmt.Errorf("mamba2: %w", err)
		}

		// Mamba2 layers still have shared MLP
		if err := loadFFNFromShards(allData, allMetadata, mapping, block, prefix); err != nil {
			return fmt.Errorf("FFN: %w", err)
		}

		// Load norms for Mamba2 layer
		inputNormKey := prefix + mapping.InputNormKey
		if inputNormKey == "" || mapping.InputNormKey == "" {
			return fmt.Errorf("input norm key is empty (prefix=%s, InputNormKey=%s)", prefix, mapping.InputNormKey)
		}
		if err := loadNormFromShards(allData, allMetadata, inputNormKey, block.AttnLN); err != nil {
			return fmt.Errorf("input norm (key=%s): %w", inputNormKey, err)
		}
		if mapping.PostAttnNormKey != "" {
			postNormKey := prefix + mapping.PostAttnNormKey
			if err := loadNormFromShards(allData, allMetadata, postNormKey, block.FFNLN); err != nil {
				return fmt.Errorf("post attn norm (key=%s): %w", postNormKey, err)
			}
		}
	} else {
		// Load attention weights
		if err := loadAttentionFromShards(allData, allMetadata, mapping, block, prefix, config); err != nil {
			return fmt.Errorf("attention: %w", err)
		}

		// Load FFN weights
		if err := loadFFNFromShards(allData, allMetadata, mapping, block, prefix); err != nil {
			return fmt.Errorf("FFN: %w", err)
		}

		// Load norms
		if config.BlockStyle == BlockParallel {
			if err := loadNormFromShards(allData, allMetadata, prefix+mapping.InputNormKey, block.InputLN); err != nil {
				return fmt.Errorf("input norm: %w", err)
			}
		} else {
			// Sequential: load two norms
			attnNormKey := mapping.AttnNormKey
			if attnNormKey == "" {
				attnNormKey = mapping.InputNormKey
			}
			if err := loadNormFromShards(allData, allMetadata, prefix+attnNormKey, block.AttnLN); err != nil {
				return fmt.Errorf("attn norm: %w", err)
			}

			if mapping.PostAttnNormKey != "" {
				if err := loadNormFromShards(allData, allMetadata, prefix+mapping.PostAttnNormKey, block.FFNLN); err != nil {
					return fmt.Errorf("FFN norm: %w", err)
				}
			} else if err := loadNormFromShards(allData, allMetadata, prefix+mapping.FFNNormKey, block.FFNLN); err != nil {
				return fmt.Errorf("FFN norm: %w", err)
			}
		}
	}

	return nil
}

// loadAttentionFromShards loads attention weights from shards
func loadAttentionFromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string, config *ModelConfig) error {
	switch attn := block.Attention.(type) {
	case *MultiQueryAttention:
		// MQA: Load combined QKV or separate
		if mapping.QKVCombined {
			var qkvWeight *Tensor
			if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionQKey, &qkvWeight); err != nil {
				return err
			}
			// Transpose BEFORE splitting if PyTorch format
			if mapping.TransposeWeights {
				qkvWeight = Transpose(qkvWeight)
			}
			// Split into Q and KV
			if mapping.QKVSplitFn != nil {
				Q, K, V := mapping.QKVSplitFn(qkvWeight, config)
				attn.QWeight = Q
				attn.KVWeight = combineMQAKV(K, V)
			}
		} else {
			loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionQKey, &attn.QWeight)
			loadTensorFromShardsOptional(allData, allMetadata, prefix+mapping.AttentionKVKey, &attn.KVWeight)
			// Transpose if PyTorch format
			if mapping.TransposeWeights {
				attn.QWeight = Transpose(attn.QWeight)
				if attn.KVWeight != nil {
					attn.KVWeight = Transpose(attn.KVWeight)
				}
			}
		}
		loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)

		// Transpose output weight if PyTorch format
		if mapping.TransposeWeights {
			attn.OutWeight = Transpose(attn.OutWeight)
		}

	case *GroupedQueryAttention:
		// GQA: Load separate Q, K, V weights
		loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionQKey, &attn.QWeight)
		loadTensorFromShards(allData, allMetadata, prefix+".self_attn.k_proj.weight", &attn.KWeight)
		loadTensorFromShards(allData, allMetadata, prefix+".self_attn.v_proj.weight", &attn.VWeight)
		loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)

		// Transpose if PyTorch format
		if mapping.TransposeWeights {
			attn.QWeight = Transpose(attn.QWeight)
			attn.KWeight = Transpose(attn.KWeight)
			attn.VWeight = Transpose(attn.VWeight)
			attn.OutWeight = Transpose(attn.OutWeight)
		}

	case *MultiHeadAttention:
		// MHA: Load combined or separate Q,K,V
		if mapping.QKVCombined {
			var qkvWeight *Tensor
			if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionQKey, &qkvWeight); err != nil {
				return err
			}
			if mapping.QKVSplitFn != nil {
				Q, K, V := mapping.QKVSplitFn(qkvWeight, config)
				attn.QWeight = Q
				attn.KWeight = K
				attn.VWeight = V
			}
			// Load and split combined QKV bias
			qBiasKey := strings.Replace(mapping.AttentionQKey, ".weight", ".bias", 1)
			var qkvBias *Tensor
			loadTensorFromShardsOptional(allData, allMetadata, prefix+qBiasKey, &qkvBias)
			if qkvBias != nil {
				hidden := config.Hidden
				attn.QBias = &Tensor{Data: qkvBias.Data[0:hidden], Shape: []int{hidden}}
				attn.KBias = &Tensor{Data: qkvBias.Data[hidden : 2*hidden], Shape: []int{hidden}}
				attn.VBias = &Tensor{Data: qkvBias.Data[2*hidden : 3*hidden], Shape: []int{hidden}}
			}
		} else {
			loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionQKey, &attn.QWeight)
			loadTensorFromShardsOptional(allData, allMetadata, prefix+".self_attn.k_proj.weight", &attn.KWeight)
			loadTensorFromShardsOptional(allData, allMetadata, prefix+".self_attn.v_proj.weight", &attn.VWeight)
			// Load separate biases
			loadTensorFromShardsOptional(allData, allMetadata, prefix+mapping.AttentionQKey+".bias", &attn.QBias)
			loadTensorFromShardsOptional(allData, allMetadata, prefix+".self_attn.k_proj.bias", &attn.KBias)
			loadTensorFromShardsOptional(allData, allMetadata, prefix+".self_attn.v_proj.bias", &attn.VBias)
		}
		loadTensorFromShards(allData, allMetadata, prefix+mapping.AttentionOutKey, &attn.OutWeight)
		outBiasKey := strings.Replace(mapping.AttentionOutKey, ".weight", ".bias", 1)
		loadTensorFromShardsOptional(allData, allMetadata, prefix+outBiasKey, &attn.OutBias)

		// Transpose if PyTorch format
		if mapping.TransposeWeights {
			attn.QWeight = Transpose(attn.QWeight)
			if attn.KWeight != nil {
				attn.KWeight = Transpose(attn.KWeight)
			}
			if attn.VWeight != nil {
				attn.VWeight = Transpose(attn.VWeight)
			}
			attn.OutWeight = Transpose(attn.OutWeight)
		}
	}

	return nil
}

// loadMamba2FromShards loads Mamba2 layer weights from shards
func loadMamba2FromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string, config *ModelConfig) error {
	mamba, ok := block.Attention.(*Mamba2Layer)
	if !ok {
		return fmt.Errorf("block does not contain Mamba2Layer")
	}

	// Load SSM parameters
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2ALogKey, &mamba.ALog); err != nil {
		return fmt.Errorf("A_log: %w", err)
	}
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2DKey, &mamba.D); err != nil {
		return fmt.Errorf("D: %w", err)
	}

	// Load conv weights
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2Conv1dWeight, &mamba.ConvWeight); err != nil {
		return fmt.Errorf("conv weight: %w", err)
	}
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2Conv1dBias, &mamba.ConvBias); err != nil {
		return fmt.Errorf("conv bias: %w", err)
	}

	// Load delta bias
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2DtBiasKey, &mamba.DeltaBias); err != nil {
		return fmt.Errorf("dt bias: %w", err)
	}

	// Load projection weights
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2InProjKey, &mamba.InProj); err != nil {
		return fmt.Errorf("in_proj: %w", err)
	}
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2NormKey, &mamba.Norm); err != nil {
		return fmt.Errorf("norm: %w", err)
	}
	if err := loadTensorFromShards(allData, allMetadata, prefix+mapping.Mamba2OutProjKey, &mamba.OutProj); err != nil {
		return fmt.Errorf("out_proj: %w", err)
	}

	// NOTE: Testing without transpose - weights might already be in correct format
	// if mapping.TransposeWeights {
	// 	mamba.InProj = Transpose(mamba.InProj)
	// 	mamba.OutProj = Transpose(mamba.OutProj)
	// }

	return nil
}

// loadFFNFromShards loads FFN weights from shards
func loadFFNFromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, mapping *WeightMapping, block *GenericBlock, prefix string) error {
	if block.FFN == nil {
		return nil
	}

	// For SwiGLU with separate gate/up projections
	if block.FFN.UseSwiGLU {
		var gateWeight, upWeight *Tensor
		gateKey := prefix + mapping.FFNUpKey
		upKey := strings.Replace(gateKey, "gate_proj", "up_proj", 1)

		errGate := loadTensorFromShards(allData, allMetadata, gateKey, &gateWeight)
		errUp := loadTensorFromShards(allData, allMetadata, upKey, &upWeight)

		if errGate == nil && errUp == nil {
			// Both found - concatenate them
			if mapping.TransposeWeights {
				gateWeight = Transpose(gateWeight)
				upWeight = Transpose(upWeight)
			}
			block.FFN.W1 = ConcatenateLastDim(gateWeight, upWeight)
		} else {
			// Fall back to single weight
			loadTensorFromShards(allData, allMetadata, prefix+mapping.FFNUpKey, &block.FFN.W1)
			if mapping.TransposeWeights {
				block.FFN.W1 = Transpose(block.FFN.W1)
			}
		}
	} else {
		// Standard FFN
		loadTensorFromShards(allData, allMetadata, prefix+mapping.FFNUpKey, &block.FFN.W1)
		if mapping.TransposeWeights {
			block.FFN.W1 = Transpose(block.FFN.W1)
		}
	}

	loadTensorFromShards(allData, allMetadata, prefix+mapping.FFNDownKey, &block.FFN.W2)
	if mapping.TransposeWeights {
		block.FFN.W2 = Transpose(block.FFN.W2)
	}

	// Biases (optional)
	loadTensorFromShardsOptional(allData, allMetadata, prefix+mapping.FFNUpKey+".bias", &block.FFN.B1)
	loadTensorFromShardsOptional(allData, allMetadata, prefix+mapping.FFNDownKey+".bias", &block.FFN.B2)

	return nil
}

// loadNormFromShards loads normalization layer weights from shards
func loadNormFromShards(allData map[string][]byte, allMetadata map[string]TensorInfo, key string, norm *LayerNormLayer) error {
	if err := loadTensorFromShards(allData, allMetadata, key, &norm.Weight); err != nil {
		return err
	}
	// Bias is optional
	biasKey := strings.Replace(key, ".weight", ".bias", 1)
	loadTensorFromShardsOptional(allData, allMetadata, biasKey, &norm.Bias)
	return nil
}
