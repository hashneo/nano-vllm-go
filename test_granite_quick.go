package main

import (
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
)

func main() {
	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		modelDir = "./models/granite-350m"
	}

	fmt.Println("Quick Granite Test - Generating 1 token")
	fmt.Println("========================================")
	fmt.Println()

	// Create config
	config := nanovllm.NewConfig(
		modelDir,
		nanovllm.WithMaxNumSeqs(1),
		nanovllm.WithMaxModelLen(512),
		nanovllm.WithMaxNumBatchedTokens(512),
	)

	// Create LLM
	llm := nanovllm.NewLLM(config)
	defer llm.Close()

	// Set up sampling parameters - 10 tokens
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.3),
		nanovllm.WithMaxTokens(10),  // Generate 10 tokens
	)

	// Simple prompt
	prompts := []string{"Q: What is 2+2?\nA:"}

	fmt.Printf("Generating 1 token for prompt: %s\n", prompts[0])
	fmt.Println()

	// Generate
	outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print results
	fmt.Println("\n✅ SUCCESS! Generation completed!")
	fmt.Println("================================")
	for i, output := range outputs {
		fmt.Printf("\nPrompt: %s\n", prompts[i])
		fmt.Printf("Output: %s\n", output.Text)
		fmt.Printf("Tokens: %d\n", len(output.TokenIDs))
	}
}
