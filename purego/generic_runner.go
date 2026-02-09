package purego

import (
	"fmt"
	"math"
	"math/rand"
	"path/filepath"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego/tensor"
)

// GenericModelRunner implements ModelRunner for any transformer architecture
// Supports GPT-2, Falcon, Llama, Mistral, and other architectures
type GenericModelRunner struct {
	model       *tensor.TransformerModel
	config      *tensor.ModelConfig
	initialized bool
}

// NewGenericModelRunner creates a model runner from a model directory
// The directory should contain:
//   - model.safetensors (model weights)
//   - config.json (HuggingFace config) OR model_info.json (our format)
func NewGenericModelRunner(modelDir string, engineConfig *nanovllm.Config) (*GenericModelRunner, error) {
	// Load model from directory (auto-detects architecture)
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	runner := &GenericModelRunner{
		model:       model,
		config:      model.Config,
		initialized: true,
	}

	fmt.Printf("✓ Loaded %s model (%s architecture)\n",
		model.Config.ModelName,
		model.Config.Architecture)

	return runner, nil
}

// NewGenericModelRunnerFromConfig creates a model runner with explicit config
// Use this if you want to specify architecture manually
func NewGenericModelRunnerFromConfig(modelPath string, modelConfig *tensor.ModelConfig, engineConfig *nanovllm.Config) (*GenericModelRunner, error) {
	// Load model with explicit config
	model, err := tensor.LoadModel(modelPath, modelConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	runner := &GenericModelRunner{
		model:       model,
		config:      modelConfig,
		initialized: true,
	}

	return runner, nil
}

// NewFalconRunner is a convenience function for Falcon models
func NewFalconRunner(modelDir string, size string, engineConfig *nanovllm.Config) (*GenericModelRunner, error) {
	config := tensor.NewFalconConfig(size)
	modelPath := filepath.Join(modelDir, "model.safetensors")
	return NewGenericModelRunnerFromConfig(modelPath, config, engineConfig)
}

// NewLlamaRunner is a convenience function for Llama models
func NewLlamaRunner(modelDir string, size string, engineConfig *nanovllm.Config) (*GenericModelRunner, error) {
	config := tensor.NewLlamaConfig(size)
	modelPath := filepath.Join(modelDir, "model.safetensors")
	return NewGenericModelRunnerFromConfig(modelPath, config, engineConfig)
}

// Run executes inference on the sequences
func (m *GenericModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	if !m.initialized {
		return nil, fmt.Errorf("model runner not initialized")
	}

	batchSize := len(seqs)
	if batchSize == 0 {
		return nil, fmt.Errorf("no sequences to process")
	}

	tokenIDs := make([]int, batchSize)

	// Process each sequence
	for i, seq := range seqs {
		if len(seq.TokenIDs) == 0 {
			return nil, fmt.Errorf("sequence %d has no tokens", seq.SeqID)
		}

		// Run model forward pass
		logits := m.model.Forward(seq.TokenIDs)

		// Get logits for last token
		lastTokenLogits := m.model.GetLogitsForLastToken(logits)

		// Sample next token
		tokenID := m.sampleToken(lastTokenLogits, seq.Temperature)
		tokenIDs[i] = tokenID
	}

	return tokenIDs, nil
}

// sampleToken samples a token from logits using temperature sampling
func (m *GenericModelRunner) sampleToken(logits []float32, temperature float64) int {
	// Make a copy
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	// Apply temperature
	if temperature > 0 && temperature != 1.0 {
		for i := range logitsCopy {
			logitsCopy[i] /= float32(temperature)
		}
	}

	// Greedy decoding for temperature 0
	if temperature == 0 {
		maxIdx := 0
		maxVal := logitsCopy[0]
		for i, val := range logitsCopy {
			if val > maxVal {
				maxVal = val
				maxIdx = i
			}
		}
		return maxIdx
	}

	// Compute softmax
	maxLogit := logitsCopy[0]
	for _, logit := range logitsCopy {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float32
	probs := make([]float32, len(logitsCopy))
	for i, logit := range logitsCopy {
		probs[i] = float32(math.Exp(float64(logit - maxLogit)))
		sumExp += probs[i]
	}

	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample from distribution
	r := rand.Float32()
	var cumProb float32
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}

	return len(probs) - 1
}

// Close cleans up resources
func (m *GenericModelRunner) Close() error {
	m.initialized = false
	return nil
}

// GetVocabSize returns the vocabulary size
func (m *GenericModelRunner) GetVocabSize() int {
	return m.config.VocabSize
}

// SetVocabSize sets the vocabulary size (not used for generic runner)
func (m *GenericModelRunner) SetVocabSize(size int) {
	// Ignored - vocab size comes from model config
}

// GetModelConfig returns the model configuration
func (m *GenericModelRunner) GetModelConfig() *tensor.ModelConfig {
	return m.config
}
