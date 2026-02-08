package nanovllm

import (
	"testing"
)

func TestSequenceCreation(t *testing.T) {
	samplingParams := NewSamplingParams(
		WithTemperature(0.8),
		WithMaxTokens(100),
	)

	tokenIDs := []int{1, 2, 3, 4, 5}
	seq := NewSequence(tokenIDs, samplingParams)

	if seq.Len() != 5 {
		t.Errorf("Expected length 5, got %d", seq.Len())
	}

	if seq.NumPromptTokens != 5 {
		t.Errorf("Expected 5 prompt tokens, got %d", seq.NumPromptTokens)
	}

	if seq.NumCompletionTokens() != 0 {
		t.Errorf("Expected 0 completion tokens, got %d", seq.NumCompletionTokens())
	}

	if seq.Status != StatusWaiting {
		t.Errorf("Expected status WAITING, got %v", seq.Status)
	}
}

func TestSequenceAppendToken(t *testing.T) {
	samplingParams := NewSamplingParams()
	tokenIDs := []int{1, 2, 3}
	seq := NewSequence(tokenIDs, samplingParams)

	seq.AppendToken(4)

	if seq.Len() != 4 {
		t.Errorf("Expected length 4, got %d", seq.Len())
	}

	if seq.LastToken != 4 {
		t.Errorf("Expected last token 4, got %d", seq.LastToken)
	}

	if seq.NumCompletionTokens() != 1 {
		t.Errorf("Expected 1 completion token, got %d", seq.NumCompletionTokens())
	}
}

func TestSequenceBlocks(t *testing.T) {
	samplingParams := NewSamplingParams()
	tokenIDs := make([]int, 600) // More than 2 blocks
	for i := range tokenIDs {
		tokenIDs[i] = i
	}
	seq := NewSequence(tokenIDs, samplingParams)

	numBlocks := seq.NumBlocks()
	expectedBlocks := 3 // 600 / 256 = 2.34, rounded up to 3
	if numBlocks != expectedBlocks {
		t.Errorf("Expected %d blocks, got %d", expectedBlocks, numBlocks)
	}

	// Test block retrieval
	block0 := seq.Block(0)
	if len(block0) != 256 {
		t.Errorf("Expected block 0 to have 256 tokens, got %d", len(block0))
	}

	block2 := seq.Block(2)
	expectedLastBlockSize := 600 - 2*256
	if len(block2) != expectedLastBlockSize {
		t.Errorf("Expected last block to have %d tokens, got %d", expectedLastBlockSize, len(block2))
	}
}

func TestSamplingParams(t *testing.T) {
	sp := NewSamplingParams(
		WithTemperature(0.7),
		WithMaxTokens(128),
		WithIgnoreEOS(true),
	)

	if sp.Temperature != 0.7 {
		t.Errorf("Expected temperature 0.7, got %f", sp.Temperature)
	}

	if sp.MaxTokens != 128 {
		t.Errorf("Expected max tokens 128, got %d", sp.MaxTokens)
	}

	if !sp.IgnoreEOS {
		t.Errorf("Expected ignore EOS to be true")
	}
}

func TestSamplingParamsValidation(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic for invalid temperature")
		}
	}()

	// This should panic due to temperature being too low
	NewSamplingParams(WithTemperature(0.0))
}
