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
// Handles special tokens (like Llama 3's <|begin_of_text|>) properly
func (t *UniversalTokenizer) Encode(text string) ([]int, error) {
	tokens := make([]int, 0)

	// Process text, looking for special tokens first
	i := 0
	for i < len(text) {
		// Check if we're at a special token
		foundSpecial := false
		for specialToken, tokenID := range t.addedTokens {
			if strings.HasPrefix(text[i:], specialToken) {
				tokens = append(tokens, tokenID)
				i += len(specialToken)
				foundSpecial = true
				break
			}
		}

		if foundSpecial {
			continue
		}

		// Not a special token, collect until next special token or end
		nextSpecialPos := len(text)
		for specialToken := range t.addedTokens {
			pos := strings.Index(text[i:], specialToken)
			if pos >= 0 && pos < nextSpecialPos {
				nextSpecialPos = pos
			}
		}

		// Extract the non-special text chunk
		end := i + nextSpecialPos
		if end > len(text) {
			end = len(text)
		}
		chunk := text[i:end]

		if chunk != "" {
			// Process the chunk with simple tokenization
			chunkTokens := t.encodeChunk(chunk)
			tokens = append(tokens, chunkTokens...)
		}

		i = end
	}

	return tokens, nil
}

// encodeChunk encodes a text chunk that doesn't contain special tokens
func (t *UniversalTokenizer) encodeChunk(text string) []int {
	// For BPE tokenizers (like Llama), spaces are encoded as Ġ (U+0120)
	// Convert spaces to Ġ for vocab lookup
	tokens := make([]int, 0)

	// Process character by character, treating space as a prefix marker
	i := 0
	for i < len(text) {
		// Start building a token
		var token strings.Builder

		// Skip and handle whitespace
		if i < len(text) && (text[i] == ' ' || text[i] == '\n' || text[i] == '\t') {
			// For newlines/tabs, encode as special token if in vocab
			if text[i] == '\n' {
				if id, ok := t.vocab["\n"]; ok {
					tokens = append(tokens, id)
				} else if id, ok := t.vocab["Ċ"]; ok { // BPE newline marker
					tokens = append(tokens, id)
				}
				i++
				continue
			} else if text[i] == '\t' {
				if id, ok := t.vocab["\t"]; ok {
					tokens = append(tokens, id)
				}
				i++
				continue
			}
			// Space - skip it, next token will have Ġ prefix
			i++
			if i >= len(text) {
				break
			}
			token.WriteRune('Ġ') // U+0120 - BPE space marker
		}

		// Collect the word/token
		start := i
		for i < len(text) && text[i] != ' ' && text[i] != '\n' && text[i] != '\t' {
			i++
		}

		if i > start {
			token.WriteString(text[start:i])
		}

		if token.Len() == 0 {
			continue
		}

		tokenStr := token.String()

		// Try exact match
		if id, ok := t.vocab[tokenStr]; ok {
			tokens = append(tokens, id)
		} else {
			// Try without Ġ prefix
			tokenStrNoSpace := strings.TrimPrefix(tokenStr, "Ġ")
			if id, ok := t.vocab[tokenStrNoSpace]; ok {
				tokens = append(tokens, id)
			} else {
				// Character-level fallback
				for _, ch := range tokenStr {
					if ch == 'Ġ' {
						continue // Skip the space marker
					}
					charStr := string(ch)
					if id, ok := t.vocab[charStr]; ok {
						tokens = append(tokens, id)
					} else {
						tokens = append(tokens, 0) // Unknown
					}
				}
			}
		}
	}

	return tokens
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

	text := result.String()

	// Apply byte-level decoding (used by GPT-2, Llama, Granite, Falcon)
	// Converts Unicode range U+0100-U+01FF back to bytes 0x00-0xFF
	// Common mappings: Ġ (U+0120) → space (0x20), Ċ (U+010A) → newline (0x0A)
	text = decodeByteLevelBPE(text)

	return text, nil
}

// decodeByteLevelBPE converts byte-level BPE encoding back to normal text
// Byte-level BPE (used by GPT-2, Llama, Granite, Falcon) maps bytes to Unicode:
// - Printable ASCII (0x21-0x7E): stays as-is (! to ~)
// - Special bytes (0x00-0x20, 0x7F-0xFF): encoded as U+0100 + byte_value
// Common mappings: Ġ (U+0120) → space, Ċ (U+010A) → newline, ĉ (U+0109) → tab
func decodeByteLevelBPE(text string) string {
	var result []byte

	for _, r := range text {
		// Byte-level encoded characters (U+0100 to U+01FF)
		// These represent bytes 0x00-0xFF that were encoded
		if r >= 0x0100 && r <= 0x01FF {
			// Convert back: U+0100 → 0x00, U+010A → 0x0A (newline), U+0120 → 0x20 (space)
			result = append(result, byte(r-0x0100))
		} else if r < 0x0100 {
			// Regular printable ASCII (already correct) or Latin-1
			result = append(result, byte(r))
		} else {
			// Other Unicode (shouldn't normally appear in byte-level BPE but handle gracefully)
			result = append(result, []byte(string(r))...)
		}
	}

	return string(result)
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
