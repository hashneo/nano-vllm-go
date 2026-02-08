package nanovllm

import "sync/atomic"

// SequenceStatus represents the status of a sequence
type SequenceStatus int

const (
	StatusWaiting SequenceStatus = iota
	StatusRunning
	StatusFinished
)

// Sequence represents a single generation request
type Sequence struct {
	SeqID           int64
	Status          SequenceStatus
	TokenIDs        []int
	LastToken       int
	NumTokens       int
	NumPromptTokens int
	NumCachedTokens int
	BlockTable      []int
	Temperature     float64
	MaxTokens       int
	IgnoreEOS       bool
	BlockSize       int
}

var seqCounter int64 = 0

// NewSequence creates a new sequence from token IDs and sampling parameters
func NewSequence(tokenIDs []int, samplingParams *SamplingParams) *Sequence {
	seqID := atomic.AddInt64(&seqCounter, 1) - 1

	// Make a copy of token IDs
	tokens := make([]int, len(tokenIDs))
	copy(tokens, tokenIDs)

	return &Sequence{
		SeqID:           seqID,
		Status:          StatusWaiting,
		TokenIDs:        tokens,
		LastToken:       tokenIDs[len(tokenIDs)-1],
		NumTokens:       len(tokenIDs),
		NumPromptTokens: len(tokenIDs),
		NumCachedTokens: 0,
		BlockTable:      make([]int, 0),
		Temperature:     samplingParams.Temperature,
		MaxTokens:       samplingParams.MaxTokens,
		IgnoreEOS:       samplingParams.IgnoreEOS,
		BlockSize:       256, // Default block size
	}
}

// Len returns the number of tokens in the sequence
func (s *Sequence) Len() int {
	return s.NumTokens
}

// IsFinished returns true if the sequence has finished generating
func (s *Sequence) IsFinished() bool {
	return s.Status == StatusFinished
}

// NumCompletionTokens returns the number of completion tokens
func (s *Sequence) NumCompletionTokens() int {
	return s.NumTokens - s.NumPromptTokens
}

// PromptTokenIDs returns the prompt token IDs
func (s *Sequence) PromptTokenIDs() []int {
	return s.TokenIDs[:s.NumPromptTokens]
}

// CompletionTokenIDs returns the completion token IDs
func (s *Sequence) CompletionTokenIDs() []int {
	return s.TokenIDs[s.NumPromptTokens:]
}

// NumCachedBlocks returns the number of cached blocks
func (s *Sequence) NumCachedBlocks() int {
	return s.NumCachedTokens / s.BlockSize
}

// NumBlocks returns the total number of blocks needed
func (s *Sequence) NumBlocks() int {
	return (s.NumTokens + s.BlockSize - 1) / s.BlockSize
}

// LastBlockNumTokens returns the number of tokens in the last block
func (s *Sequence) LastBlockNumTokens() int {
	return s.NumTokens - (s.NumBlocks()-1)*s.BlockSize
}

// Block returns the tokens in the i-th block
func (s *Sequence) Block(i int) []int {
	if i < 0 || i >= s.NumBlocks() {
		return nil
	}
	start := i * s.BlockSize
	end := (i + 1) * s.BlockSize
	if end > len(s.TokenIDs) {
		end = len(s.TokenIDs)
	}
	return s.TokenIDs[start:end]
}

// AppendToken appends a token to the sequence
func (s *Sequence) AppendToken(tokenID int) {
	s.TokenIDs = append(s.TokenIDs, tokenID)
	s.LastToken = tokenID
	s.NumTokens++
}
