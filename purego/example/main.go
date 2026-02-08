package main

import (
	"fmt"
	"log"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Nano-vLLM-Go - Pure Go Example")
	fmt.Println("================================")
	fmt.Println()

	// Configuration
	modelPath := "./model.onnx"        // Path to ONNX model
	tokenizerPath := "./tokenizer.json" // Path to tokenizer

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(128),
		nanovllm.WithMaxNumBatchedTokens(8192),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithTensorParallelSize(1),
		nanovllm.WithEOS(2),
	)

	// Create ONNX model runner
	fmt.Println("Loading ONNX model...")
	modelRunner, err := purego.NewONNXModelRunner(modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create model runner: %v", err)
	}
	defer modelRunner.Close()

	// Create tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := purego.NewHFTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}
	defer tokenizer.Close()

	// Update config with actual EOS token
	config.EOS = tokenizer.EOSTokenID()

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(100),
	)

	// Generate
	prompts := []string{
		"Hello, how are you?",
		"What is the capital of France?",
		"Explain quantum computing in simple terms.",
	}

	fmt.Println("Generating responses...")
	fmt.Println()

	outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}

	// Print results
	fmt.Println("\nResults:")
	fmt.Println("========")
	for i, output := range outputs {
		fmt.Printf("\nPrompt %d: %s\n", i+1, prompts[i])
		fmt.Printf("Output: %s\n", output.Text)
		fmt.Printf("Tokens: %d\n", len(output.TokenIDs))
	}
}
