package purego

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// HFTokenizer implements a simple HuggingFace tokenizer loader
type HFTokenizer struct {
	vocab       map[string]int
	invVocab    map[int]string
	eosID       int
	bosID       int
	padID       int
	vocabSize   int
	addedTokens map[string]int
}

// TokenizerConfig represents the tokenizer_config.json structure
type TokenizerConfig struct {
	EOSToken string `json:"eos_token"`
	BOSToken string `json:"bos_token"`
	PadToken string `json:"pad_token"`
}

// TokenizerJSON represents the tokenizer.json structure
type TokenizerJSON struct {
	Model struct {
		Vocab map[string]int `json:"vocab"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int    `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// NewHFTokenizer creates a new HuggingFace tokenizer from a directory
func NewHFTokenizer(tokenizerDir string) (*HFTokenizer, error) {
	tokenizer := &HFTokenizer{
		vocab:       make(map[string]int),
		invVocab:    make(map[int]string),
		addedTokens: make(map[string]int),
		eosID:       -1,
		bosID:       -1,
		padID:       -1,
	}

	// Load tokenizer.json
	tokenizerPath := filepath.Join(tokenizerDir, "tokenizer.json")
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json: %w", err)
	}

	var tokenizerJSON TokenizerJSON
	if err := json.Unmarshal(data, &tokenizerJSON); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer.json: %w", err)
	}

	// Load vocab
	for token, id := range tokenizerJSON.Model.Vocab {
		tokenizer.vocab[token] = id
		tokenizer.invVocab[id] = token
	}

	// Load added tokens
	for _, addedToken := range tokenizerJSON.AddedTokens {
		tokenizer.addedTokens[addedToken.Content] = addedToken.ID
		tokenizer.vocab[addedToken.Content] = addedToken.ID
		tokenizer.invVocab[addedToken.ID] = addedToken.Content
	}

	tokenizer.vocabSize = len(tokenizer.invVocab)

	// Load tokenizer_config.json for special tokens
	configPath := filepath.Join(tokenizerDir, "tokenizer_config.json")
	configData, err := os.ReadFile(configPath)
	if err == nil {
		var config TokenizerConfig
		if err := json.Unmarshal(configData, &config); err == nil {
			if id, ok := tokenizer.vocab[config.EOSToken]; ok {
				tokenizer.eosID = id
			}
			if id, ok := tokenizer.vocab[config.BOSToken]; ok {
				tokenizer.bosID = id
			}
			if id, ok := tokenizer.vocab[config.PadToken]; ok {
				tokenizer.padID = id
			}
		}
	}

	// Load model_info.json if available (from our export script)
	infoPath := filepath.Join(tokenizerDir, "model_info.json")
	if infoData, err := os.ReadFile(infoPath); err == nil {
		var info struct {
			EOSTokenID int `json:"eos_token_id"`
			BOSTokenID int `json:"bos_token_id"`
			PadTokenID int `json:"pad_token_id"`
			VocabSize  int `json:"vocab_size"`
		}
		if err := json.Unmarshal(infoData, &info); err == nil {
			if info.EOSTokenID > 0 {
				tokenizer.eosID = info.EOSTokenID
			}
			if info.BOSTokenID > 0 {
				tokenizer.bosID = info.BOSTokenID
			}
			if info.PadTokenID > 0 {
				tokenizer.padID = info.PadTokenID
			}
			if info.VocabSize > 0 {
				tokenizer.vocabSize = info.VocabSize
			}
		}
	}

	fmt.Printf("✓ Loaded HF tokenizer (vocab: %d, EOS: %d)\n", tokenizer.vocabSize, tokenizer.eosID)
	return tokenizer, nil
}

// Encode converts text to token IDs
// Note: This is a simplified implementation
func (t *HFTokenizer) Encode(text string) ([]int, error) {
	// Very simple word-level tokenization
	// For production, use a proper BPE/WordPiece implementation
	words := strings.Fields(text)
	tokens := make([]int, 0, len(words))

	for _, word := range words {
		// Clean word
		word = strings.ToLower(word)
		word = strings.Trim(word, ".,!?;:")

		// Look up in vocab
		if id, ok := t.vocab[word]; ok {
			tokens = append(tokens, id)
		} else if id, ok := t.addedTokens[word]; ok {
			tokens = append(tokens, id)
		} else {
			// Use a fallback token (first token or 0)
			tokens = append(tokens, 0)
		}
	}

	return tokens, nil
}

// Decode converts token IDs to text
func (t *HFTokenizer) Decode(tokenIDs []int) (string, error) {
	words := make([]string, 0, len(tokenIDs))

	for _, id := range tokenIDs {
		if id == t.eosID || id == t.padID {
			continue
		}
		if token, ok := t.invVocab[id]; ok {
			words = append(words, token)
		}
	}

	return strings.Join(words, " "), nil
}

// EOSTokenID returns the EOS token ID
func (t *HFTokenizer) EOSTokenID() int {
	return t.eosID
}

// BOSTokenID returns the BOS token ID
func (t *HFTokenizer) BOSTokenID() int {
	return t.bosID
}

// VocabSize returns the vocabulary size
func (t *HFTokenizer) VocabSize() int {
	return t.vocabSize
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
