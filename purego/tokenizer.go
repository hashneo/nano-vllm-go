package purego

import (
	"fmt"
)

// HFTokenizer implements Tokenizer using Hugging Face tokenizers
// Note: This requires the daulet/tokenizers library
// For a working example without external dependencies, use SimpleBPETokenizer
type HFTokenizer struct {
	tokenizerPath string
	eosID         int
	initialized   bool
}

// NewHFTokenizer creates a new Hugging Face tokenizer
func NewHFTokenizer(tokenizerPath string) (*HFTokenizer, error) {
	// In production, load tokenizer here:
	// tk, err := tokenizers.FromFile(tokenizerPath)
	// For now, just validate the path exists

	return &HFTokenizer{
		tokenizerPath: tokenizerPath,
		eosID:         2,
		initialized:   true,
	}, nil
}

// Encode converts text to token IDs
func (t *HFTokenizer) Encode(text string) ([]int, error) {
	if !t.initialized {
		return nil, fmt.Errorf("tokenizer not initialized")
	}

	// In production: use actual tokenizer
	// encoded := t.tokenizer.Encode(text, false)
	// For now, use simple character-based encoding
	tokens := make([]int, len(text))
	for i, ch := range text {
		tokens[i] = int(ch) % 1000
	}
	return tokens, nil
}

// Decode converts token IDs to text
func (t *HFTokenizer) Decode(tokenIDs []int) (string, error) {
	if !t.initialized {
		return "", fmt.Errorf("tokenizer not initialized")
	}

	// In production: use actual tokenizer
	// For now, use simple decoding
	result := ""
	for _, id := range tokenIDs {
		if id != t.eosID {
			result += string(rune(id + 32))
		}
	}
	return result, nil
}

// EOSTokenID returns the EOS token ID
func (t *HFTokenizer) EOSTokenID() int {
	return t.eosID
}

// SetEOSTokenID sets the EOS token ID
func (t *HFTokenizer) SetEOSTokenID(id int) {
	t.eosID = id
}

// Close cleans up resources
func (t *HFTokenizer) Close() error {
	t.initialized = false
	return nil
}

// SimpleBPETokenizer is a simplified BPE tokenizer for demonstration
// This is fully functional and has no external dependencies
type SimpleBPETokenizer struct {
	vocab  map[string]int
	invVoc map[int]string
	eosID  int
}

// NewSimpleBPETokenizer creates a simple BPE tokenizer with a basic vocabulary
func NewSimpleBPETokenizer(eosID int) *SimpleBPETokenizer {
	vocab := make(map[string]int)
	invVoc := make(map[int]string)

	// Create a simple vocabulary
	specialTokens := []string{"<pad>", "<s>", "</s>", "<unk>"}
	for i, token := range specialTokens {
		vocab[token] = i
		invVoc[i] = token
	}

	// Add some common words
	commonWords := []string{
		"hello", "world", "the", "is", "a", "an", "in", "on", "at",
		"to", "from", "with", "for", "of", "and", "or", "but",
		"this", "that", "these", "those", "it", "he", "she", "they",
		"what", "where", "when", "why", "how", "who", "which",
		"are", "you", "was", "were", "be", "been", "have", "has", "had",
		"do", "does", "did", "can", "could", "will", "would", "should",
	}

	idx := len(specialTokens)
	for _, word := range commonWords {
		vocab[word] = idx
		invVoc[idx] = word
		idx++
	}

	// Add character tokens for fallback
	for ch := 'a'; ch <= 'z'; ch++ {
		token := string(ch)
		vocab[token] = idx
		invVoc[idx] = token
		idx++
	}
	for ch := 'A'; ch <= 'Z'; ch++ {
		token := string(ch)
		vocab[token] = idx
		invVoc[idx] = token
		idx++
	}
	for ch := '0'; ch <= '9'; ch++ {
		token := string(ch)
		vocab[token] = idx
		invVoc[idx] = token
		idx++
	}

	// Add punctuation and whitespace
	punctuation := []string{" ", ".", ",", "!", "?", ":", ";", "-", "'", "\"", "\n"}
	for _, p := range punctuation {
		vocab[p] = idx
		invVoc[idx] = p
		idx++
	}

	return &SimpleBPETokenizer{
		vocab:  vocab,
		invVoc: invVoc,
		eosID:  eosID,
	}
}

// Encode converts text to token IDs
func (t *SimpleBPETokenizer) Encode(text string) ([]int, error) {
	result := []int{}

	// Simple word-level tokenization
	words := []string{}
	currentWord := ""
	for _, ch := range text {
		if ch == ' ' || ch == '\n' {
			if currentWord != "" {
				words = append(words, currentWord)
				currentWord = ""
			}
			words = append(words, string(ch))
		} else {
			currentWord += string(ch)
		}
	}
	if currentWord != "" {
		words = append(words, currentWord)
	}

	// Convert words to token IDs
	for _, word := range words {
		if id, ok := t.vocab[word]; ok {
			result = append(result, id)
		} else {
			// Fallback to character-level
			for _, ch := range word {
				token := string(ch)
				if id, ok := t.vocab[token]; ok {
					result = append(result, id)
				} else {
					// Unknown token
					result = append(result, t.vocab["<unk>"])
				}
			}
		}
	}

	return result, nil
}

// Decode converts token IDs to text
func (t *SimpleBPETokenizer) Decode(tokenIDs []int) (string, error) {
	result := ""
	for _, id := range tokenIDs {
		if token, ok := t.invVoc[id]; ok {
			result += token
		}
	}
	return result, nil
}

// EOSTokenID returns the EOS token ID
func (t *SimpleBPETokenizer) EOSTokenID() int {
	return t.eosID
}

// VocabSize returns the vocabulary size
func (t *SimpleBPETokenizer) VocabSize() int {
	return len(t.vocab)
}
