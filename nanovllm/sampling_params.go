package nanovllm

import "fmt"

// SamplingParams holds the sampling parameters for generation
type SamplingParams struct {
	Temperature float64
	MaxTokens   int
	IgnoreEOS   bool
}

// SamplingOption is a functional option for SamplingParams
type SamplingOption func(*SamplingParams)

// NewSamplingParams creates a new SamplingParams with default values
func NewSamplingParams(opts ...SamplingOption) *SamplingParams {
	sp := &SamplingParams{
		Temperature: 1.0,
		MaxTokens:   64,
		IgnoreEOS:   false,
	}

	for _, opt := range opts {
		opt(sp)
	}

	if err := sp.validate(); err != nil {
		panic(err)
	}

	return sp
}

// validate checks if the sampling parameters are valid
func (sp *SamplingParams) validate() error {
	if sp.Temperature <= 1e-10 {
		return fmt.Errorf("greedy sampling is not permitted (temperature too low)")
	}
	return nil
}

// WithTemperature sets the sampling temperature
func WithTemperature(t float64) SamplingOption {
	return func(sp *SamplingParams) {
		sp.Temperature = t
	}
}

// WithMaxTokens sets the maximum number of tokens to generate
func WithMaxTokens(n int) SamplingOption {
	return func(sp *SamplingParams) {
		sp.MaxTokens = n
	}
}

// WithIgnoreEOS sets whether to ignore the EOS token
func WithIgnoreEOS(b bool) SamplingOption {
	return func(sp *SamplingParams) {
		sp.IgnoreEOS = b
	}
}
