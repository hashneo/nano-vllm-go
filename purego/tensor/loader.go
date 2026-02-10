package tensor

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unsafe"
)

// TensorInfo describes a tensor in safetensors format
type TensorInfo struct {
	Dtype  string   `json:"dtype"`
	Shape  []int    `json:"shape"`
	Offset [2]int64 `json:"data_offsets"`
}

// LoadGPT2FromSafetensors loads GPT-2 weights from safetensors file
func LoadGPT2FromSafetensors(path string, config *GPT2Config) (*GPT2Model, error) {
	// Read file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	// Parse header
	headerSize := binary.LittleEndian.Uint64(data[:8])
	headerBytes := data[8 : 8+headerSize]
	tensorData := data[8+headerSize:]

	var metadata map[string]TensorInfo
	if err := json.Unmarshal(headerBytes, &metadata); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Create model
	model := NewGPT2Model(config)

	// Load embeddings
	if err := loadTensor(tensorData, metadata, "wte.weight", &model.TokenEmbedding); err != nil {
		return nil, fmt.Errorf("failed to load token embedding: %w", err)
	}
	if err := loadTensor(tensorData, metadata, "wpe.weight", &model.PosEmbedding); err != nil {
		return nil, fmt.Errorf("failed to load position embedding: %w", err)
	}

	// Load transformer blocks
	for i := 0; i < config.NumLayers; i++ {
		// Try both naming conventions: "h.0" and "transformer.h.0"
		prefix := fmt.Sprintf("h.%d", i)
		block := model.Blocks[i]

		// Attention
		if err := loadTensor(tensorData, metadata, prefix+".attn.c_attn.weight", &block.Attention.QWeight); err == nil {
			// GPT-2 uses combined QKV weights, need to split
			splitQKV(block.Attention)
		}
		loadTensor(tensorData, metadata, prefix+".attn.c_attn.bias", &block.Attention.QBias)
		loadTensor(tensorData, metadata, prefix+".attn.c_proj.weight", &block.Attention.OutWeight)
		loadTensor(tensorData, metadata, prefix+".attn.c_proj.bias", &block.Attention.OutBias)

		// FFN
		loadTensor(tensorData, metadata, prefix+".mlp.c_fc.weight", &block.FFN.W1)
		loadTensor(tensorData, metadata, prefix+".mlp.c_fc.bias", &block.FFN.B1)
		loadTensor(tensorData, metadata, prefix+".mlp.c_proj.weight", &block.FFN.W2)
		loadTensor(tensorData, metadata, prefix+".mlp.c_proj.bias", &block.FFN.B2)

		// Layer norms
		loadTensor(tensorData, metadata, prefix+".ln_1.weight", &block.LN1.Weight)
		loadTensor(tensorData, metadata, prefix+".ln_1.bias", &block.LN1.Bias)
		loadTensor(tensorData, metadata, prefix+".ln_2.weight", &block.LN2.Weight)
		loadTensor(tensorData, metadata, prefix+".ln_2.bias", &block.LN2.Bias)
	}

	// Final layer norm
	model.LNFinal = &LayerNormLayer{Eps: 1e-5}
	loadTensor(tensorData, metadata, "ln_f.weight", &model.LNFinal.Weight)
	loadTensor(tensorData, metadata, "ln_f.bias", &model.LNFinal.Bias)

	// LM head (usually tied with token embedding)
	model.LMHead = Transpose(model.TokenEmbedding)

	fmt.Printf("✓ Loaded GPT-2 weights from %s\n", path)
	model.PrintInfo()

	return model, nil
}

// loadTensor loads a single tensor from safetensors data
func loadTensor(data []byte, metadata map[string]TensorInfo, name string, target **Tensor) error {
	info, ok := metadata[name]
	if !ok {
		// Try with "transformer." prefix
		altName := "transformer." + name
		info, ok = metadata[altName]
		if !ok {
			// Try replacing dots with underscores
			altName = strings.ReplaceAll(name, ".", "_")
			info, ok = metadata[altName]
			if !ok {
				return fmt.Errorf("tensor not found: %s (tried: %s, transformer.%s)", name, altName, name)
			}
		}
	}

	// Extract tensor data
	start := info.Offset[0]
	end := info.Offset[1]
	tensorBytes := data[start:end]

	// Convert to float32
	numElements := 1
	for _, dim := range info.Shape {
		numElements *= dim
	}

	tensorData := make([]float32, numElements)

	switch info.Dtype {
	case "F32":
		// Direct copy
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(tensorBytes[i*4 : (i+1)*4])
			tensorData[i] = float32frombits(bits)
		}
	case "F16":
		// Convert from float16
		for i := 0; i < numElements; i++ {
			bits := binary.LittleEndian.Uint16(tensorBytes[i*2 : (i+1)*2])
			tensorData[i] = float32fromfloat16(bits)
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

// splitQKV splits combined QKV weight into separate Q, K, V
func splitQKV(attn *MultiHeadAttention) {
	// GPT-2 stores Q,K,V as one big matrix [hidden, 3*hidden]
	// We need to split it into separate Q, K, V matrices
	if attn.QWeight == nil || len(attn.QWeight.Shape) != 2 {
		return
	}

	hidden := attn.QWeight.Shape[0]
	combined := attn.QWeight.Shape[1]

	if combined != 3*hidden {
		return // Already split
	}

	// Save original combined data BEFORE splitting
	combinedData := attn.QWeight.Data

	// Extract Q, K, V from the combined data
	attn.QWeight = &Tensor{
		Data:  combinedData[0 : hidden*hidden],
		Shape: []int{hidden, hidden},
	}
	attn.KWeight = &Tensor{
		Data:  combinedData[hidden*hidden : 2*hidden*hidden],
		Shape: []int{hidden, hidden},
	}
	attn.VWeight = &Tensor{
		Data:  combinedData[2*hidden*hidden : 3*hidden*hidden],
		Shape: []int{hidden, hidden},
	}

	// Split bias if present
	if attn.QBias != nil && len(attn.QBias.Data) == 3*hidden {
		// Save original combined bias BEFORE splitting
		combinedBias := attn.QBias.Data

		attn.QBias = &Tensor{
			Data:  combinedBias[0:hidden],
			Shape: []int{hidden},
		}
		attn.KBias = &Tensor{
			Data:  combinedBias[hidden : 2*hidden],
			Shape: []int{hidden},
		}
		attn.VBias = &Tensor{
			Data:  combinedBias[2*hidden : 3*hidden],
			Shape: []int{hidden},
		}
	}
}

// Helper functions for type conversion
func float32frombits(bits uint32) float32 {
	ptr := &bits
	return *(*float32)(unsafe.Pointer(ptr))
}

func float32fromfloat16(bits uint16) float32 {
	// Simple float16 to float32 conversion
	sign := uint32((bits >> 15) & 1)
	exp := uint32((bits >> 10) & 0x1F)
	frac := uint32(bits & 0x3FF)

	if exp == 0 {
		if frac == 0 {
			// Zero
			return float32frombits(sign << 31)
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
	return float32frombits(result)
}
