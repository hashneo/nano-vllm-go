package purego

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"

	"nano-vllm-go/nanovllm"
)

// HTTPModelRunner implements ModelRunner using HTTP calls to a Python server
type HTTPModelRunner struct {
	serverURL string
	client    *http.Client
	vocabSize int
}

// NewHTTPModelRunner creates a new HTTP-based model runner
func NewHTTPModelRunner(serverURL string) (*HTTPModelRunner, error) {
	runner := &HTTPModelRunner{
		serverURL: serverURL,
		client:    &http.Client{},
	}

	// Get model info
	resp, err := runner.client.Get(serverURL + "/info")
	if err != nil {
		return nil, fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	var info struct {
		VocabSize  int    `json:"vocab_size"`
		EOSTokenID int    `json:"eos_token_id"`
		ModelType  string `json:"model_type"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("failed to decode server info: %w", err)
	}

	runner.vocabSize = info.VocabSize
	fmt.Printf("✓ Connected to model (vocab: %d)\n", info.VocabSize)

	return runner, nil
}

// Run executes inference via HTTP
func (m *HTTPModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	type SeqData struct {
		TokenIDs    []int   `json:"token_ids"`
		Temperature float64 `json:"temperature"`
	}

	type Request struct {
		Sequences []SeqData `json:"sequences"`
		IsPrefill bool      `json:"is_prefill"`
	}

	req := Request{
		Sequences: make([]SeqData, len(seqs)),
		IsPrefill: isPrefill,
	}

	for i, seq := range seqs {
		req.Sequences[i] = SeqData{
			TokenIDs:    seq.TokenIDs,
			Temperature: seq.Temperature,
		}
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := m.client.Post(m.serverURL+"/inference", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		TokenIDs []int `json:"token_ids"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.TokenIDs, nil
}

// Close cleans up resources
func (m *HTTPModelRunner) Close() error {
	return nil
}

// SetVocabSize sets the vocabulary size
func (m *HTTPModelRunner) SetVocabSize(size int) {
	m.vocabSize = size
}

// HTTPTokenizer implements Tokenizer using HTTP calls
type HTTPTokenizer struct {
	serverURL string
	eosID     int
}

// NewHTTPTokenizer creates a new HTTP-based tokenizer
func NewHTTPTokenizer(serverURL string, eosID int) *HTTPTokenizer {
	return &HTTPTokenizer{
		serverURL: serverURL,
		eosID:     eosID,
	}
}

// Encode converts text to token IDs via HTTP
func (t *HTTPTokenizer) Encode(text string) ([]int, error) {
	type Request struct {
		Text string `json:"text"`
	}

	req := Request{Text: text}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	resp, err := http.Post(t.serverURL+"/tokenize", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result struct {
		Tokens []int `json:"tokens"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result.Tokens, nil
}

// Decode converts token IDs to text via HTTP
func (t *HTTPTokenizer) Decode(tokenIDs []int) (string, error) {
	type Request struct {
		Tokens []int `json:"tokens"`
	}

	req := Request{Tokens: tokenIDs}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}

	resp, err := http.Post(t.serverURL+"/detokenize", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	var result struct {
		Text string `json:"text"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	return result.Text, nil
}

// EOSTokenID returns the EOS token ID
func (t *HTTPTokenizer) EOSTokenID() int {
	return t.eosID
}
