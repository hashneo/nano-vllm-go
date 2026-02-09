package purego

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// UniversalTokenizer loads HuggingFace tokenizers from any model
type UniversalTokenizer struct {
	vocab       map[string]int
	invVocab    map[int]string
	eosID       int
	bosID       int
	padID       int
	vocabSize   int
	addedTokens map[string]int
	modelType   string
}

// NewUniversalTokenizer creates a tokenizer from any HuggingFace model directory
func NewUniversalTokenizer(tokenizerDir string) (*UniversalTokenizer, error) {
	tokenizer := &UniversalTokenizer{
		vocab:       make(map[string]int),
		invVocab:    make(map[int]string),
		addedTokens: make(map[string]int),
		eosID:       -1,
		bosID:       -1,
		padID:       -1,
	}

	// Try tokenizer.json first (most common)
	if err := tokenizer.loadFromTokenizerJSON(tokenizerDir); err != nil {
		// Fall back to vocab.json (GPT-2 style)
		if err := tokenizer.loadFromVocabJSON(tokenizerDir); err != nil {
			return nil, fmt.Errorf("failed to load tokenizer: %w", err)
		}
	}

	// Load special tokens from tokenizer_config.json
	tokenizer.loadTokenizerConfig(tokenizerDir)

	// Load from config.json if available
	tokenizer.loadModelConfig(tokenizerDir)

	// Load from model_info.json if available (our custom format)
	tokenizer.loadModelInfo(tokenizerDir)

	fmt.Printf("✓ Loaded tokenizer (vocab: %d, EOS: %d, BOS: %d)\n",
		tokenizer.vocabSize, tokenizer.eosID, tokenizer.bosID)

	return tokenizer, nil
}

// loadFromTokenizerJSON loads from tokenizer.json (standard HuggingFace format)
func (t *UniversalTokenizer) loadFromTokenizerJSON(dir string) error {
	path := filepath.Join(dir, "tokenizer.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var tokenizerJSON struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &tokenizerJSON); err != nil {
		return err
	}

	// Load vocab
	for token, id := range tokenizerJSON.Model.Vocab {
		t.vocab[token] = id
		t.invVocab[id] = token
	}

	// Load added tokens
	for _, addedToken := range tokenizerJSON.AddedTokens {
		t.addedTokens[addedToken.Content] = addedToken.ID
		t.vocab[addedToken.Content] = addedToken.ID
		t.invVocab[addedToken.ID] = addedToken.Content
	}

	t.vocabSize = len(t.invVocab)
	return nil
}

// loadFromVocabJSON loads from vocab.json (GPT-2 style)
func (t *UniversalTokenizer) loadFromVocabJSON(dir string) error {
	path := filepath.Join(dir, "vocab.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, &t.vocab); err != nil {
		return err
	}

	// Create inverse mapping
	for token, id := range t.vocab {
		t.invVocab[id] = token
	}

	t.vocabSize = len(t.vocab)
	return nil
}

// loadTokenizerConfig loads special tokens from tokenizer_config.json
func (t *UniversalTokenizer) loadTokenizerConfig(dir string) {
	path := filepath.Join(dir, "tokenizer_config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var config struct {
		EOSToken interface{} `json:"eos_token"`
		BOSToken interface{} `json:"bos_token"`
		PadToken interface{} `json:"pad_token"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return
	}

	// Extract token strings (can be string or dict with "content" field)
	eosToken := extractTokenString(config.EOSToken)
	bosToken := extractTokenString(config.BOSToken)
	padToken := extractTokenString(config.PadToken)

	// Look up IDs
	if id, ok := t.vocab[eosToken]; ok {
		t.eosID = id
	}
	if id, ok := t.vocab[bosToken]; ok {
		t.bosID = id
	}
	if id, ok := t.vocab[padToken]; ok {
		t.padID = id
	}
}

// loadModelConfig loads from config.json (HuggingFace model config)
func (t *UniversalTokenizer) loadModelConfig(dir string) {
	path := filepath.Join(dir, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var config struct {
		VocabSize  int    `json:"vocab_size"`
		EOSTokenID int    `json:"eos_token_id"`
		BOSTokenID int    `json:"bos_token_id"`
		PadTokenID int    `json:"pad_token_id"`
		ModelType  string `json:"model_type"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return
	}

	if config.VocabSize > 0 {
		t.vocabSize = config.VocabSize
	}
	if config.EOSTokenID >= 0 {
		t.eosID = config.EOSTokenID
	}
	if config.BOSTokenID >= 0 {
		t.bosID = config.BOSTokenID
	}
	if config.PadTokenID >= 0 {
		t.padID = config.PadTokenID
	}
	if config.ModelType != "" {
		t.modelType = config.ModelType
	}
}

// loadModelInfo loads from model_info.json (our custom format)
func (t *UniversalTokenizer) loadModelInfo(dir string) {
	path := filepath.Join(dir, "model_info.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var info struct {
		VocabSize  int `json:"vocab_size"`
		EOSTokenID int `json:"eos_token_id"`
		BOSTokenID int `json:"bos_token_id"`
		PadTokenID int `json:"pad_token_id"`
	}

	if err := json.Unmarshal(data, &info); err != nil {
		return
	}

	if info.VocabSize > 0 {
		t.vocabSize = info.VocabSize
	}
	if info.EOSTokenID >= 0 {
		t.eosID = info.EOSTokenID
	}
	if info.BOSTokenID >= 0 {
		t.bosID = info.BOSTokenID
	}
	if info.PadTokenID >= 0 {
		t.padID = info.PadTokenID
	}
}

// extractTokenString extracts token string from JSON value (can be string or dict)
func extractTokenString(val interface{}) string {
	switch v := val.(type) {
	case string:
		return v
	case map[string]interface{}:
		if content, ok := v["content"].(string); ok {
			return content
		}
	}
	return ""
}

// Encode converts text to token IDs
// Note: This is a simplified word-level tokenization
// For production, integrate with actual BPE/WordPiece tokenizer
func (t *UniversalTokenizer) Encode(text string) ([]int, error) {
	// Simple word-level tokenization
	words := t.simpleTokenize(text)
	tokens := make([]int, 0, len(words))

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
					charStr := string(ch)
					if id, ok := t.vocab[charStr]; ok {
						tokens = append(tokens, id)
					} else {
						// Use first token as unknown
						tokens = append(tokens, 0)
					}
				}
			}
		}
	}

	return tokens, nil
}

// simpleTokenize splits text into tokens
func (t *UniversalTokenizer) simpleTokenize(text string) []string {
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
func (t *UniversalTokenizer) Decode(tokenIDs []int) (string, error) {
	var result strings.Builder

	for _, id := range tokenIDs {
		if id == t.eosID || id == t.padID {
			break
		}
		if token, ok := t.invVocab[id]; ok {
			result.WriteString(token)
		}
	}

	return result.String(), nil
}

// EOSTokenID returns the EOS token ID
func (t *UniversalTokenizer) EOSTokenID() int {
	return t.eosID
}

// BOSTokenID returns the BOS token ID
func (t *UniversalTokenizer) BOSTokenID() int {
	return t.bosID
}

// VocabSize returns the vocabulary size
func (t *UniversalTokenizer) VocabSize() int {
	return t.vocabSize
}

// ModelType returns the detected model type
func (t *UniversalTokenizer) ModelType() string {
	return t.modelType
}
