package purego

import (
	"fmt"
	"math"
	"math/rand"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego/tensor"
)

// NativeModelRunner implements ModelRunner using pure Go transformer
type NativeModelRunner struct {
	model       *tensor.GPT2Model
	initialized bool
}

// NewNativeModelRunner creates a new pure Go model runner
func NewNativeModelRunner(modelPath string, config *nanovllm.Config) (*NativeModelRunner, error) {
	// Infer GPT-2 config from model
	// For now, hardcode GPT-2 small config
	gpt2Config := &tensor.GPT2Config{
		VocabSize:  50257,
		Hidden:     768,
		NumLayers:  12,
		NumHeads:   12,
		FFNDim:     3072,
		MaxSeqLen:  1024,
		EOSTokenID: 50256,
	}

	// Load model weights
	model, err := tensor.LoadGPT2FromSafetensors(modelPath, gpt2Config)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	runner := &NativeModelRunner{
		model:       model,
		initialized: true,
	}

	return runner, nil
}

// Run executes inference on the sequences
func (m *NativeModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	if !m.initialized {
		return nil, fmt.Errorf("model runner not initialized")
	}

	batchSize := len(seqs)
	if batchSize == 0 {
		return nil, fmt.Errorf("no sequences to process")
	}

	tokenIDs := make([]int, batchSize)

	// Process each sequence (for now, one at a time)
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
func (m *NativeModelRunner) sampleToken(logits []float32, temperature float64) int {
	// Make a copy
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	// Apply temperature
	if temperature != 1.0 {
		for i := range logitsCopy {
			logitsCopy[i] /= float32(temperature)
		}
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
func (m *NativeModelRunner) Close() error {
	m.initialized = false
	return nil
}

// GetVocabSize returns the vocabulary size
func (m *NativeModelRunner) GetVocabSize() int {
	return m.model.Config.VocabSize
}

// SetVocabSize sets the vocabulary size (not used for native)
func (m *NativeModelRunner) SetVocabSize(size int) {
	// Ignored - vocab size comes from model
}
