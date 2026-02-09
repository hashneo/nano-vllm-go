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

	fmt.Println("Quick Granite Test - 20 tokens max")
	fmt.Println("====================================")
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

	// Set up sampling parameters - 20 tokens (practical for demos)
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(20), // Generate 20 tokens
	)

	// Get prompt from command line or use default
	prompt := "Q: What is 2+2?\nA:"
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}
	prompts := []string{prompt}

	fmt.Printf("Generating response for: %s\n", prompts[0])
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
