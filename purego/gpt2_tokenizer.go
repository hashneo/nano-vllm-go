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
	vocab       map[string]int
	invVocab    map[int]string
	eosID       int
	bosID       int
	byteDecoder map[rune]byte
}

// buildByteDecoder creates the GPT-2 byte-to-unicode decoder
// GPT-2 uses a special mapping to make all bytes valid UTF-8
func buildByteDecoder() map[rune]byte {
	// Build the reverse of GPT-2's bytes_to_unicode mapping
	byteEncoder := make(map[byte]rune)
	byteDecoder := make(map[rune]byte)

	// Visible ASCII characters
	for b := byte('!'); b <= byte('~'); b++ {
		byteEncoder[b] = rune(b)
	}
	for b := byte('¡'); b <= byte('¬'); b++ {
		byteEncoder[b] = rune(b)
	}
	// Use int to avoid byte overflow at 255
	for b := int('®'); b <= int('ÿ'); b++ {
		byteEncoder[byte(b)] = rune(b)
	}

	// Fill in the remaining bytes with special Unicode characters
	n := 0
	for b := 0; b < 256; b++ {
		if _, ok := byteEncoder[byte(b)]; !ok {
			byteEncoder[byte(b)] = rune(256 + n)
			n++
		}
	}

	// Create reverse mapping
	for b, r := range byteEncoder {
		byteDecoder[r] = b
	}

	return byteDecoder
}

// NewGPT2Tokenizer creates a GPT-2 tokenizer from vocab files
func NewGPT2Tokenizer(vocabDir string) (*GPT2Tokenizer, error) {
	tokenizer := &GPT2Tokenizer{
		vocab:       make(map[string]int),
		invVocab:    make(map[int]string),
		eosID:       50256, // GPT-2 <|endoftext|>
		bosID:       50256,
		byteDecoder: buildByteDecoder(),
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
	// Collect token strings
	var tokenStrings []string
	for _, id := range tokenIDs {
		if id == t.eosID {
			break
		}
		if token, ok := t.invVocab[id]; ok {
			tokenStrings = append(tokenStrings, token)
		}
	}

	// Join all tokens
	text := strings.Join(tokenStrings, "")

	// Decode byte-level encoding to actual bytes
	var bytes []byte
	for _, r := range text {
		if b, ok := t.byteDecoder[r]; ok {
			bytes = append(bytes, b)
		} else {
			// If not in decoder, assume it's a normal character
			bytes = append(bytes, byte(r))
		}
	}

	// Convert bytes to UTF-8 string
	return string(bytes), nil
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
