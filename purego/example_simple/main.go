package main

import (
	"fmt"
	"log"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Nano-vLLM-Go - Simple BPE Example")
	fmt.Println("===================================")
	fmt.Println()

	// This example uses the simple BPE tokenizer (no external files needed)
	// and demonstrates the architecture without requiring ONNX models

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(64),
		nanovllm.WithMaxNumBatchedTokens(4096),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithEOS(2),
	)

	// Create simple BPE tokenizer
	tokenizer := purego.NewSimpleBPETokenizer(2)
	fmt.Printf("Vocabulary size: %d\n", tokenizer.VocabSize())

	// Create mock model runner (for demonstration)
	modelRunner := nanovllm.NewMockModelRunner(config)

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.8),
		nanovllm.WithMaxTokens(50),
	)

	// Test tokenization first
	fmt.Println("\nTesting tokenization:")
	testText := "hello world this is a test"
	tokens, err := tokenizer.Encode(testText)
	if err != nil {
		log.Fatalf("Encoding failed: %v", err)
	}
	fmt.Printf("  Input: %s\n", testText)
	fmt.Printf("  Tokens: %v\n", tokens)

	decoded, err := tokenizer.Decode(tokens)
	if err != nil {
		log.Fatalf("Decoding failed: %v", err)
	}
	fmt.Printf("  Decoded: %s\n", decoded)

	// Generate
	prompts := []string{
		"hello world",
		"how are you",
		"this is a test",
	}

	fmt.Println("\nGenerating responses...")
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

	fmt.Println("\nNote: This example uses a simple tokenizer and mock model.")
	fmt.Println("For real inference, use the ONNX example with actual models.")
}
