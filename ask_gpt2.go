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

	// Set sampling parameters (temperature, top-p, top-k)
	modelRunner.SetSamplingParams(0.8, 0.95, 50)

	// Create GPT-2 tokenizer
	tokenizer, err := purego.NewGPT2Tokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v\n", err)
	}

	// Create LLM with real model
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	// With KV caching, generation is now O(N) - much faster!
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.8),
		nanovllm.WithMaxTokens(30), // Generate 30 tokens (fast with KV cache!)
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

	// Check for repetitive pattern
	if len(outputs[0].TokenIDs) > 3 {
		allSame := true
		firstID := outputs[0].TokenIDs[0]
		for _, id := range outputs[0].TokenIDs[1:] {
			if id != firstID {
				allSame = false
				break
			}
		}
		if allSame {
			fmt.Println("⚠️  WARNING: All tokens are identical!")
		} else {
			fmt.Println("✓ Tokens are varied - sampling is working!")
		}
	}
	fmt.Println(strings.Repeat("=", 50))
}
