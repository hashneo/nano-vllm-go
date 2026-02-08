package nanovllm

import (
	"fmt"
	"time"

	"github.com/schollz/progressbar/v3"
)

// Output represents the output of a generation request
type Output struct {
	Text     string
	TokenIDs []int
}

// LLMEngine is the main inference engine
type LLMEngine struct {
	config      *Config
	modelRunner ModelRunner
	tokenizer   Tokenizer
	scheduler   *Scheduler
}

// NewLLMEngine creates a new LLM engine
func NewLLMEngine(config *Config, modelRunner ModelRunner, tokenizer Tokenizer) *LLMEngine {
	return &LLMEngine{
		config:      config,
		modelRunner: modelRunner,
		tokenizer:   tokenizer,
		scheduler:   NewScheduler(config),
	}
}

// Close cleans up resources
func (e *LLMEngine) Close() error {
	return e.modelRunner.Close()
}

// AddRequest adds a generation request to the engine
func (e *LLMEngine) AddRequest(prompt interface{}, samplingParams *SamplingParams) error {
	var tokenIDs []int
	var err error

	switch p := prompt.(type) {
	case string:
		tokenIDs, err = e.tokenizer.Encode(p)
		if err != nil {
			return fmt.Errorf("failed to encode prompt: %w", err)
		}
	case []int:
		tokenIDs = p
	default:
		return fmt.Errorf("prompt must be string or []int")
	}

	seq := NewSequence(tokenIDs, samplingParams)
	e.scheduler.Add(seq)
	return nil
}

// Step performs one inference step
func (e *LLMEngine) Step() ([]Output, int, error) {
	seqs, isPrefill := e.scheduler.Schedule()

	tokenIDs, err := e.modelRunner.Run(seqs, isPrefill)
	if err != nil {
		return nil, 0, fmt.Errorf("model inference failed: %w", err)
	}

	e.scheduler.Postprocess(seqs, tokenIDs)

	outputs := make([]Output, 0)
	for _, seq := range seqs {
		if seq.IsFinished() {
			text, err := e.tokenizer.Decode(seq.CompletionTokenIDs())
			if err != nil {
				return nil, 0, fmt.Errorf("failed to decode tokens: %w", err)
			}
			outputs = append(outputs, Output{
				Text:     text,
				TokenIDs: seq.CompletionTokenIDs(),
			})
		}
	}

	// Calculate number of tokens processed
	numTokens := 0
	if isPrefill {
		for _, seq := range seqs {
			numTokens += seq.Len()
		}
	} else {
		numTokens = -len(seqs) // Negative for decode phase
	}

	return outputs, numTokens, nil
}

// IsFinished returns true if all requests have been processed
func (e *LLMEngine) IsFinished() bool {
	return e.scheduler.IsFinished()
}

// Generate generates completions for the given prompts
func (e *LLMEngine) Generate(prompts []interface{}, samplingParams interface{}, useTqdm bool) ([]Output, error) {
	// Convert sampling params
	var spList []*SamplingParams
	switch sp := samplingParams.(type) {
	case *SamplingParams:
		spList = make([]*SamplingParams, len(prompts))
		for i := range spList {
			spList[i] = sp
		}
	case []*SamplingParams:
		if len(sp) != len(prompts) {
			return nil, fmt.Errorf("number of sampling params must match number of prompts")
		}
		spList = sp
	default:
		return nil, fmt.Errorf("samplingParams must be *SamplingParams or []*SamplingParams")
	}

	// Add all requests
	for i, prompt := range prompts {
		if err := e.AddRequest(prompt, spList[i]); err != nil {
			return nil, err
		}
	}

	// Set up progress bar
	var bar *progressbar.ProgressBar
	if useTqdm {
		bar = progressbar.NewOptions(len(prompts),
			progressbar.OptionSetDescription("Generating"),
			progressbar.OptionSetWidth(40),
			progressbar.OptionShowCount(),
			progressbar.OptionShowIts(),
			progressbar.OptionSetTheme(progressbar.Theme{
				Saucer:        "=",
				SaucerHead:    ">",
				SaucerPadding: " ",
				BarStart:      "[",
				BarEnd:        "]",
			}),
		)
	}

	outputMap := make(map[int64][]int)
	var prefillThroughput, decodeThroughput float64

	for !e.IsFinished() {
		start := time.Now()
		stepOutputs, numTokens, err := e.Step()
		if err != nil {
			return nil, err
		}
		elapsed := time.Since(start).Seconds()

		if useTqdm {
			if numTokens > 0 {
				prefillThroughput = float64(numTokens) / elapsed
			} else {
				decodeThroughput = float64(-numTokens) / elapsed
			}
			bar.Describe(fmt.Sprintf("Generating [Prefill: %dtok/s, Decode: %dtok/s]",
				int(prefillThroughput), int(decodeThroughput)))
		}

		for _, output := range stepOutputs {
			// Find the sequence ID (we need to track this better)
			// For now, use the order of completion
			for i := int64(0); i < int64(len(prompts)); i++ {
				if _, exists := outputMap[i]; !exists {
					outputMap[i] = output.TokenIDs
					if useTqdm {
						bar.Add(1)
					}
					break
				}
			}
		}
	}

	if useTqdm {
		bar.Finish()
	}

	// Reconstruct outputs in order
	outputs := make([]Output, len(prompts))
	for i := range prompts {
		tokenIDs := outputMap[int64(i)]
		text, err := e.tokenizer.Decode(tokenIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to decode tokens: %w", err)
		}
		outputs[i] = Output{
			Text:     text,
			TokenIDs: tokenIDs,
		}
	}

	return outputs, nil
}
