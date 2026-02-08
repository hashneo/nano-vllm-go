package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

// ModelConfig holds the model configuration
type ModelConfig struct {
	VocabSize     int    `json:"vocab_size"`
	EOSTokenID    int    `json:"eos_token_id"`
	PadTokenID    int    `json:"pad_token_id"`
	ModelPath     string `json:"model_path"`
	TokenizerPath string `json:"tokenizer_path"`
}

func loadModelConfig(path string) (*ModelConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var config ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

func main() {
	fmt.Println("Nano-vLLM-Go - Real Model Test")
	fmt.Println("================================")
	fmt.Println()

	// Load model config
	configPath := os.Getenv("MODEL_CONFIG")
	if configPath == "" {
		configPath = "./models/onnx/nano_config.json"
	}

	fmt.Printf("Loading model config from: %s\n", configPath)
	modelConfig, err := loadModelConfig(configPath)
	if err != nil {
		log.Fatalf("Failed to load model config: %v\n", err)
	}

	fmt.Printf("✓ Model config loaded\n")
	fmt.Printf("  Vocab size: %d\n", modelConfig.VocabSize)
	fmt.Printf("  EOS token: %d\n", modelConfig.EOSTokenID)
	fmt.Printf("  Model: %s\n", modelConfig.ModelPath)
	fmt.Println()

	// Create engine config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(64),
		nanovllm.WithMaxNumBatchedTokens(4096),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithEOS(modelConfig.EOSTokenID),
	)

	// Create model runner
	fmt.Println("Loading ONNX model...")
	modelRunner, err := purego.NewONNXModelRunner(modelConfig.ModelPath, config)
	if err != nil {
		log.Fatalf("Failed to create model runner: %v\n", err)
	}
	defer modelRunner.Close()
	modelRunner.SetVocabSize(modelConfig.VocabSize)

	fmt.Println("✓ Model loaded")

	// Create tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := purego.NewHFTokenizer(modelConfig.TokenizerPath)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v\n", err)
	}

	fmt.Println("✓ Tokenizer loaded")
	fmt.Println()

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(100),
	)

	// Test questions
	questions := []string{
		"What is the capital of France?",
		"Explain quantum computing in one sentence.",
		"What is 2 + 2?",
	}

	// Allow custom question from command line
	if len(os.Args) > 1 {
		questions = os.Args[1:]
	}

	fmt.Println("Generating responses...")
	fmt.Printf("Questions: %d\n", len(questions))
	fmt.Println()

	outputs, err := llm.GenerateSimple(questions, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print results
	separator := "═══════════════════════════════════════════════════════════════════"
	fmt.Println("\n" + separator)
	fmt.Println("RESULTS")
	fmt.Println(separator)

	for i, output := range outputs {
		fmt.Printf("\n📝 Question %d: %s\n", i+1, questions[i])
		fmt.Printf("💬 Answer: %s\n", output.Text)
		fmt.Printf("📊 Tokens: %d\n", len(output.TokenIDs))
	}

	fmt.Println("\n" + separator)
	fmt.Println("✓ Test complete!")
}
