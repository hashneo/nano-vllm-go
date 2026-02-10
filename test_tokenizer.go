package main

import (
	"fmt"
	"log"

	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Testing BPE Tokenizer")
	fmt.Println("=====================\n")

	// Load tokenizer
	tokenizer, err := purego.NewBPETokenizer("./models/gpt2-small")
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v\n", err)
	}

	// Test sentences
	tests := []string{
		"Hello world",
		"What is the capital of France?",
		"Paris",
		"The cat sat on the mat",
	}

	for _, text := range tests {
		fmt.Printf("Input:  %q\n", text)

		// Encode
		tokenIDs, err := tokenizer.Encode(text)
		if err != nil {
			fmt.Printf("  Error encoding: %v\n\n", err)
			continue
		}
		fmt.Printf("Tokens: %v\n", tokenIDs)

		// Decode
		decoded, err := tokenizer.Decode(tokenIDs)
		if err != nil {
			fmt.Printf("  Error decoding: %v\n\n", err)
			continue
		}
		fmt.Printf("Output: %q\n", decoded)

		// Check round-trip
		if decoded == text {
			fmt.Println("✓ Round-trip successful!")
		} else {
			fmt.Println("⚠️  Round-trip mismatch")
		}
		fmt.Println()
	}
}
