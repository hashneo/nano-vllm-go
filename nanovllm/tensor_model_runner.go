package nanovllm

import (
	"fmt"
	"sync"

	"nano-vllm-go/purego/tensor"
)

// TensorModelRunner implements ModelRunner using the purego tensor model
type TensorModelRunner struct {
	model    *tensor.TransformerModel
	kvCaches map[int64]*tensor.KVCache // Per-sequence KV cache
	mu       sync.Mutex                // Protects kvCaches map

	// Sampling parameters (can be overridden per request)
	defaultSampling *tensor.SamplingParams
}

// NewTensorModelRunner creates a new tensor model runner
func NewTensorModelRunner(modelDir string) (*TensorModelRunner, error) {
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		return nil, fmt.Errorf("failed to load model: %w", err)
	}

	return &TensorModelRunner{
		model:           model,
		kvCaches:        make(map[int64]*tensor.KVCache),
		defaultSampling: tensor.DefaultSamplingParams(),
	}, nil
}

// SetSamplingParams sets default sampling parameters
func (m *TensorModelRunner) SetSamplingParams(temperature float32, topP float32, topK int) {
	m.defaultSampling = &tensor.SamplingParams{
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		RepetitionPenalty: 1.2, // Default repetition penalty
	}
}

// SetSamplingParamsWithRepetition sets default sampling parameters including repetition penalty
func (m *TensorModelRunner) SetSamplingParamsWithRepetition(temperature float32, topP float32, topK int, repetitionPenalty float32) {
	m.defaultSampling = &tensor.SamplingParams{
		Temperature:       temperature,
		TopP:              topP,
		TopK:              topK,
		RepetitionPenalty: repetitionPenalty,
	}
}

// Run executes model inference on the given sequences
func (m *TensorModelRunner) Run(seqs []*Sequence, isPrefill bool) ([]int, error) {
	tokenIDs := make([]int, len(seqs))

	for i, seq := range seqs {
		m.mu.Lock()

		// Get or create KV cache for this sequence
		kvCache, exists := m.kvCaches[seq.SeqID]
		if !exists || isPrefill {
			// Create new cache for prefill
			kvCache = tensor.NewKVCache(m.model.Config.NumLayers)
			m.kvCaches[seq.SeqID] = kvCache
		}
		m.mu.Unlock()

		var logits *tensor.Tensor
		var newCache *tensor.KVCache

		if isPrefill {
			// Prefill: process all tokens at once
			logits, newCache = m.model.ForwardWithCache(seq.TokenIDs, nil, 0)
		} else {
			// Decode: only process the last token
			lastToken := []int{seq.TokenIDs[len(seq.TokenIDs)-1]}
			posOffset := len(seq.TokenIDs) - 1
			logits, newCache = m.model.ForwardWithCache(lastToken, kvCache, posOffset)
		}

		// Update cache
		m.mu.Lock()
		m.kvCaches[seq.SeqID] = newCache
		m.mu.Unlock()

		// Get logits for last token
		lastTokenLogits := m.model.GetLogitsForLastToken(logits)

		// Sample using temperature/top-p/top-k with repetition penalty
		// Pass the token history to apply repetition penalty
		tokenIDs[i] = tensor.SampleWithHistory(lastTokenLogits, seq.TokenIDs, m.defaultSampling)
	}

	return tokenIDs, nil
}

// ClearCache removes KV cache for a specific sequence
func (m *TensorModelRunner) ClearCache(seqID int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.kvCaches, seqID)
}

// ClearAllCaches removes all KV caches
func (m *TensorModelRunner) ClearAllCaches() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.kvCaches = make(map[int64]*tensor.KVCache)
}

// Close cleans up resources
func (m *TensorModelRunner) Close() error {
	m.ClearAllCaches()
	return nil
}
