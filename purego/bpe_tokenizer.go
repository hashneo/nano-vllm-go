package purego

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// BPETokenizer implements GPT-2 BPE tokenization
type BPETokenizer struct {
	encoder     map[string]int
	decoder     map[int]string
	bpeRanks    map[string]int // Merge rules priority
	byteEncoder map[byte]rune
	byteDecoder map[rune]byte
	pattern     *regexp.Regexp
	eosID       int
}

// NewBPETokenizer creates a proper GPT-2 BPE tokenizer
func NewBPETokenizer(tokenizerDir string) (*BPETokenizer, error) {
	t := &BPETokenizer{
		encoder:     make(map[string]int),
		decoder:     make(map[int]string),
		bpeRanks:    make(map[string]int),
		byteEncoder: buildByteEncoder(),
		eosID:       50256,
	}

	// Build byte decoder
	t.byteDecoder = make(map[rune]byte)
	for b, r := range t.byteEncoder {
		t.byteDecoder[r] = b
	}

	// Load vocab
	vocabPath := filepath.Join(tokenizerDir, "vocab.json")
	data, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vocab: %w", err)
	}

	if err := json.Unmarshal(data, &t.encoder); err != nil {
		return nil, fmt.Errorf("failed to parse vocab: %w", err)
	}

	// Build decoder
	for token, id := range t.encoder {
		t.decoder[id] = token
	}

	// Load merges
	mergesPath := filepath.Join(tokenizerDir, "merges.txt")
	if err := t.loadMerges(mergesPath); err != nil {
		return nil, fmt.Errorf("failed to load merges: %w", err)
	}

	// GPT-2 tokenization pattern (simplified for Go's RE2)
	// Matches contractions, words, numbers, punctuation, whitespace
	t.pattern = regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

	fmt.Printf("✓ Loaded BPE tokenizer (vocab: %d, merges: %d)\n", len(t.encoder), len(t.bpeRanks))
	return t, nil
}

// buildByteEncoder creates GPT-2's byte-to-unicode mapping
func buildByteEncoder() map[byte]rune {
	encoder := make(map[byte]rune)

	// Visible ASCII
	for b := byte('!'); b <= byte('~'); b++ {
		encoder[b] = rune(b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		encoder[byte(b)] = rune(b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		encoder[byte(b)] = rune(b)
	}

	// Map remaining bytes to special Unicode range
	n := 0
	for b := 0; b < 256; b++ {
		if _, ok := encoder[byte(b)]; !ok {
			encoder[byte(b)] = rune(256 + n)
			n++
		}
	}

	return encoder
}

// loadMerges loads BPE merge rules
func (t *BPETokenizer) loadMerges(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	rank := 0

	// Skip first line (#version)
	if scanner.Scan() {
		// skip
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		t.bpeRanks[line] = rank
		rank++
	}

	return scanner.Err()
}

// Encode converts text to token IDs
func (t *BPETokenizer) Encode(text string) ([]int, error) {
	var tokenIDs []int

	// Find all matches using the pattern
	matches := t.pattern.FindAllString(text, -1)

	for _, token := range matches {
		// Convert to bytes then to BPE characters
		var bpeToken strings.Builder
		for _, b := range []byte(token) {
			bpeToken.WriteRune(t.byteEncoder[b])
		}

		// Apply BPE merges
		word := t.bpe(bpeToken.String())

		// Convert to token IDs
		for _, bpeToken := range strings.Split(word, " ") {
			if id, ok := t.encoder[bpeToken]; ok {
				tokenIDs = append(tokenIDs, id)
			}
		}
	}

	return tokenIDs, nil
}

// bpe applies byte pair encoding to a word
func (t *BPETokenizer) bpe(token string) string {
	if len(token) <= 1 {
		return token
	}

	// Convert to pairs
	word := []string{}
	for _, r := range token {
		word = append(word, string(r))
	}

	// Get all pairs
	for {
		pairs := t.getPairs(word)
		if len(pairs) == 0 {
			break
		}

		// Find pair with lowest rank (highest priority merge)
		minPair := ""
		minRank := int(^uint(0) >> 1) // Max int

		for pair := range pairs {
			if rank, ok := t.bpeRanks[pair]; ok {
				if rank < minRank {
					minRank = rank
					minPair = pair
				}
			}
		}

		if minPair == "" {
			break
		}

		// Merge the pair
		parts := strings.Split(minPair, " ")
		if len(parts) != 2 {
			break
		}

		first, second := parts[0], parts[1]
		newWord := []string{}
		i := 0

		for i < len(word) {
			// Find next occurrence of first
			j := -1
			for k := i; k < len(word); k++ {
				if word[k] == first {
					j = k
					break
				}
			}

			if j == -1 {
				newWord = append(newWord, word[i:]...)
				break
			}

			newWord = append(newWord, word[i:j]...)

			if j < len(word)-1 && word[j+1] == second {
				newWord = append(newWord, first+second)
				i = j + 2
			} else {
				newWord = append(newWord, word[j])
				i = j + 1
			}
		}

		word = newWord
	}

	return strings.Join(word, " ")
}

// getPairs gets all adjacent pairs in a word
func (t *BPETokenizer) getPairs(word []string) map[string]struct{} {
	pairs := make(map[string]struct{})
	for i := 0; i < len(word)-1; i++ {
		pair := word[i] + " " + word[i+1]
		pairs[pair] = struct{}{}
	}
	return pairs
}

// Decode converts token IDs to text
func (t *BPETokenizer) Decode(tokenIDs []int) (string, error) {
	// Get token strings
	var tokens []string
	for _, id := range tokenIDs {
		if id == t.eosID {
			break
		}
		if token, ok := t.decoder[id]; ok {
			tokens = append(tokens, token)
		}
	}

	// Join tokens
	text := strings.Join(tokens, "")

	// Decode bytes
	var bytes []byte
	for _, r := range text {
		if b, ok := t.byteDecoder[r]; ok {
			bytes = append(bytes, b)
		}
	}

	return string(bytes), nil
}

// EOSTokenID returns the EOS token ID
func (t *BPETokenizer) EOSTokenID() int {
	return t.eosID
}
