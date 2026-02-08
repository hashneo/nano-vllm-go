package main

import (
	"fmt"
	"log"

	"github.com/your-username/nano-vllm-go/nanovllm"
)

func main() {
	// Create a config (using current directory as model path for demo)
	// In production, this would be a real model directory
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(512),
		nanovllm.WithMaxNumBatchedTokens(16384),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithTensorParallelSize(1),
	)

	// Create LLM engine
	llm := nanovllm.NewLLM(config)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.6),
		nanovllm.WithMaxTokens(256),
	)

	// Prompts to generate
	prompts := []string{
		"Hello, Nano-vLLM-Go!",
		"What is the meaning of life?",
		"Explain quantum computing in simple terms.",
	}

	fmt.Println("Starting generation...")
	fmt.Println()

	// Generate outputs
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
