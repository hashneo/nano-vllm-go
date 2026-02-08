package nanovllm

// ModelRunner is an interface for running model inference
// This can be implemented using various backends:
// - CGo bindings to PyTorch/ONNX
// - Go ML libraries
// - HTTP/gRPC calls to inference servers
// - Custom CUDA kernels
type ModelRunner interface {
	// Run executes model inference on the given sequences
	// Returns the next token ID for each sequence
	Run(seqs []*Sequence, isPrefill bool) ([]int, error)

	// Close cleans up resources
	Close() error
}

// MockModelRunner is a simple mock implementation for demonstration
type MockModelRunner struct {
	config *Config
	vocab  int
}

// NewMockModelRunner creates a new mock model runner
func NewMockModelRunner(config *Config) *MockModelRunner {
	return &MockModelRunner{
		config: config,
		vocab:  32000, // Default vocab size
	}
}

// Run generates mock output tokens
func (m *MockModelRunner) Run(seqs []*Sequence, isPrefill bool) ([]int, error) {
	tokenIDs := make([]int, len(seqs))

	for i, seq := range seqs {
		// Simple mock: generate tokens based on sequence ID and position
		// In a real implementation, this would run the actual model
		tokenID := int((seq.SeqID + int64(seq.NumTokens)) % int64(m.vocab))

		// Occasionally generate EOS for testing
		if seq.NumCompletionTokens() > 10 && seq.NumCompletionTokens()%20 == 0 {
			tokenID = m.config.EOS
		}

		tokenIDs[i] = tokenID
	}

	return tokenIDs, nil
}

// Close cleans up resources
func (m *MockModelRunner) Close() error {
	return nil
}

// Tokenizer is an interface for tokenizing text
// This should be implemented using actual tokenizers like:
// - BPE (Byte Pair Encoding)
// - SentencePiece
// - Hugging Face tokenizers via CGo
type Tokenizer interface {
	// Encode converts text to token IDs
	Encode(text string) ([]int, error)

	// Decode converts token IDs to text
	Decode(tokenIDs []int) (string, error)

	// EOSTokenID returns the EOS token ID
	EOSTokenID() int
}

// MockTokenizer is a simple mock tokenizer for demonstration
type MockTokenizer struct {
	eosTokenID int
}

// NewMockTokenizer creates a new mock tokenizer
func NewMockTokenizer(eosTokenID int) *MockTokenizer {
	return &MockTokenizer{
		eosTokenID: eosTokenID,
	}
}

// Encode performs mock tokenization
func (t *MockTokenizer) Encode(text string) ([]int, error) {
	// Simple mock: convert each character to a token
	tokens := make([]int, len(text))
	for i, c := range text {
		tokens[i] = int(c) % 1000
	}
	return tokens, nil
}

// Decode performs mock detokenization
func (t *MockTokenizer) Decode(tokenIDs []int) (string, error) {
	// Simple mock: convert tokens to characters
	result := ""
	for _, id := range tokenIDs {
		if id != t.eosTokenID {
			result += string(rune(id + 32))
		}
	}
	return result, nil
}

// EOSTokenID returns the EOS token ID
func (t *MockTokenizer) EOSTokenID() int {
	return t.eosTokenID
}
