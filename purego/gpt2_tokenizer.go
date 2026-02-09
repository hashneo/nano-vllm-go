package purego

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// GPT2Tokenizer implements a simple GPT-2 tokenizer
type GPT2Tokenizer struct {
	vocab    map[string]int
	invVocab map[int]string
	eosID    int
	bosID    int
}

// NewGPT2Tokenizer creates a GPT-2 tokenizer from vocab files
func NewGPT2Tokenizer(vocabDir string) (*GPT2Tokenizer, error) {
	tokenizer := &GPT2Tokenizer{
		vocab:    make(map[string]int),
		invVocab: make(map[int]string),
		eosID:    50256, // GPT-2 <|endoftext|>
		bosID:    50256,
	}

	// Try to load vocab.json
	vocabPath := filepath.Join(vocabDir, "vocab.json")
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		// Fallback: create minimal vocab
		return tokenizer.createMinimalVocab(), nil
	}

	// Parse vocab
	if err := json.Unmarshal(data, &tokenizer.vocab); err != nil {
		return nil, fmt.Errorf("failed to parse vocab: %w", err)
	}

	// Create inverse mapping
	for token, id := range tokenizer.vocab {
		tokenizer.invVocab[id] = token
	}

	fmt.Printf("✓ Loaded GPT-2 tokenizer (vocab: %d)\n", len(tokenizer.vocab))
	return tokenizer, nil
}

// createMinimalVocab creates a minimal working vocabulary
func (t *GPT2Tokenizer) createMinimalVocab() *GPT2Tokenizer {
	// Add common tokens
	tokens := []string{
		"<|endoftext|>", " ", "the", "a", "an", "is", "are", "was", "were",
		"be", "been", "being", "have", "has", "had", "do", "does", "did",
		"will", "would", "should", "can", "could", "may", "might",
		"I", "you", "he", "she", "it", "we", "they",
		"this", "that", "these", "those", "what", "which", "who", "where", "when",
		"hello", "world", "yes", "no", "please", "thank", "sorry",
		".", ",", "!", "?", ":", ";", "-", "'", "\"", "\n",
	}

	// Add single characters
	for ch := 'a'; ch <= 'z'; ch++ {
		tokens = append(tokens, string(ch))
	}
	for ch := 'A'; ch <= 'Z'; ch++ {
		tokens = append(tokens, string(ch))
	}
	for ch := '0'; ch <= '9'; ch++ {
		tokens = append(tokens, string(ch))
	}

	// Create vocab
	for i, token := range tokens {
		t.vocab[token] = i
		t.invVocab[i] = token
	}

	// Fill remaining with <unk>
	for i := len(tokens); i < 50257; i++ {
		t.invVocab[i] = "<unk>"
	}

	t.eosID = 0 // <|endoftext|>
	fmt.Printf("✓ Created minimal GPT-2 tokenizer (vocab: %d)\n", len(tokens))
	return t
}

// Encode converts text to token IDs (simplified)
func (t *GPT2Tokenizer) Encode(text string) ([]int, error) {
	// Very simple word-level tokenization
	// For production, use proper BPE
	tokens := []int{}

	// Split on whitespace and punctuation
	words := t.simpleTokenize(text)

	for _, word := range words {
		if id, ok := t.vocab[word]; ok {
			tokens = append(tokens, id)
		} else {
			// Try lowercase
			word = strings.ToLower(word)
			if id, ok := t.vocab[word]; ok {
				tokens = append(tokens, id)
			} else {
				// Character fallback
				for _, ch := range word {
					if id, ok := t.vocab[string(ch)]; ok {
						tokens = append(tokens, id)
					} else {
						// Unknown token
						tokens = append(tokens, 1) // <unk>
					}
				}
			}
		}
	}

	return tokens, nil
}

// simpleTokenize splits text into tokens
func (t *GPT2Tokenizer) simpleTokenize(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, ch := range text {
		if ch == ' ' || ch == '\n' || ch == '\t' {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(ch))
		} else if strings.ContainsRune(".,!?:;-'\"", ch) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(ch))
		} else {
			current.WriteRune(ch)
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

// Decode converts token IDs to text
func (t *GPT2Tokenizer) Decode(tokenIDs []int) (string, error) {
	var result strings.Builder

	for _, id := range tokenIDs {
		if id == t.eosID {
			break
		}
		if token, ok := t.invVocab[id]; ok {
			result.WriteString(token)
		}
	}

	return result.String(), nil
}

// EOSTokenID returns the EOS token ID
func (t *GPT2Tokenizer) EOSTokenID() int {
	return t.eosID
}

// VocabSize returns the vocabulary size
func (t *GPT2Tokenizer) VocabSize() int {
	return 50257 // GPT-2 vocab size
}

// LoadFromFile loads tokenizer from a text file (simple format)
func (t *GPT2Tokenizer) LoadFromFile(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := 0
	for scanner.Scan() {
		token := scanner.Text()
		t.vocab[token] = id
		t.invVocab[id] = token
		id++
	}

	return scanner.Err()
}
