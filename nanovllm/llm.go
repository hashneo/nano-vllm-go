package nanovllm

// LLM is the user-facing API for the inference engine
type LLM struct {
	*LLMEngine
}

// NewLLM creates a new LLM with default components
func NewLLM(config *Config) *LLM {
	// Set up EOS token
	if config.EOS == -1 {
		config.EOS = 2 // Default EOS token
	}

	// Create tokenizer
	tokenizer := NewMockTokenizer(config.EOS)

	// Create model runner
	modelRunner := NewMockModelRunner(config)

	// Create engine
	engine := NewLLMEngine(config, modelRunner, tokenizer)

	return &LLM{
		LLMEngine: engine,
	}
}

// NewLLMWithComponents creates a new LLM with custom components
func NewLLMWithComponents(config *Config, modelRunner ModelRunner, tokenizer Tokenizer) *LLM {
	engine := NewLLMEngine(config, modelRunner, tokenizer)
	return &LLM{
		LLMEngine: engine,
	}
}

// GenerateSimple is a convenience method for generating from string prompts
func (llm *LLM) GenerateSimple(prompts []string, samplingParams *SamplingParams, useTqdm bool) ([]Output, error) {
	promptsInterface := make([]interface{}, len(prompts))
	for i, p := range prompts {
		promptsInterface[i] = p
	}
	return llm.Generate(promptsInterface, samplingParams, useTqdm)
}
