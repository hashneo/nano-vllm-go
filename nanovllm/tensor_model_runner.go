package nanovllm

import (
	"fmt"

	"nano-vllm-go/purego/tensor"
)

// TensorModelRunner implements ModelRunner using the purego tensor model
type TensorModelRunner struct {
	model *tensor.TransformerModel
}

// NewTensorModelRunner creates a new tensor model runner
func NewTensorModelRunner(modelDir string) (*TensorModelRunner, error) {
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &TensorModelRunner{
		model: model,
	}, nil
}

// Run executes model inference on the given sequences
func (m *TensorModelRunner) Run(seqs []*Sequence, isPrefill bool) ([]int, error) {
	tokenIDs := make([]int, len(seqs))

	for i, seq := range seqs {
		// Get all token IDs for this sequence
		allTokens := seq.TokenIDs

		// Run forward pass
		logits := m.model.Forward(allTokens)

		// Get logits for last token
		lastTokenLogits := m.model.GetLogitsForLastToken(logits)

		// Find token with highest logit (greedy sampling for now)
		maxIdx := 0
		maxVal := lastTokenLogits[0]
		for j := 1; j < len(lastTokenLogits); j++ {
			if lastTokenLogits[j] > maxVal {
				maxVal = lastTokenLogits[j]
				maxIdx = j
			}
		}

		tokenIDs[i] = maxIdx
	}

	return tokenIDs, nil
}

// Close cleans up resources
func (m *TensorModelRunner) Close() error {
	// No resources to clean up currently
	return nil
}
