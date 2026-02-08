//go:build pytorch
// +build pytorch

package main

import (
	"fmt"
	"log"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/pytorch"
)

func main() {
	fmt.Println("Nano-vLLM-Go - PyTorch Implementation")
	fmt.Println("======================================")
	fmt.Println()

	// Configuration
	modelPath := "./model.pt"  // TorchScript model
	tokenizerPath := "Qwen/Qwen2-0.5B"

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(128),
		nanovllm.WithMaxNumBatchedTokens(8192),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithEOS(151643),
	)

	// Load PyTorch model
	fmt.Println("Loading PyTorch model...")
	modelRunner, err := pytorch.NewPyTorchModelRunner(modelPath, config)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer modelRunner.Close()

	// Load tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := pytorch.NewPyTorchTokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}
	defer tokenizer.Close()

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(256),
	)

	// Generate
	prompts := []string{
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
