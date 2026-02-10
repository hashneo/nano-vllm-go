package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	modelDir := "./models/gpt2-small"

	// Get question from command line or use default
	question := "The capital of France is"
	if len(os.Args) > 1 {
		question = os.Args[1]
	}

	fmt.Println("GPT-2 Question Answering")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("Question: %s\n\n", question)

	// Create config
	config := nanovllm.NewConfig(
		modelDir,
		nanovllm.WithMaxNumSeqs(1),
		nanovllm.WithMaxModelLen(512),
		nanovllm.WithMaxNumBatchedTokens(512),
	)

	// Create real model runner
	modelRunner, err := nanovllm.NewTensorModelRunner(modelDir)
	if err != nil {
		log.Fatalf("Failed to create model runner: %v\n", err)
	}

	// Create GPT-2 tokenizer
	tokenizer, err := purego.NewGPT2Tokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v\n", err)
	}

	// Create LLM with real model
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	// Note: Keep max_tokens low (5-10) because generation is O(N²) without KV caching
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(5), // Generate 5 tokens (fast for testing)
	)

	// Generate
	fmt.Println("Generating response...")
	outputs, err := llm.GenerateSimple([]string{question}, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print result
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("Answer: %s\n", outputs[0].Text)
	fmt.Printf("\nGenerated %d tokens\n", len(outputs[0].TokenIDs))
	fmt.Println(strings.Repeat("=", 50))
}
