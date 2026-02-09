#!/bin/bash
# Quick Granite question script with limited tokens for practical use

if [ -z "$1" ]; then
    echo "Usage: ./ask_granite.sh \"Your question here\""
    echo "Example: ./ask_granite.sh \"What is AI?\""
    exit 1
fi

# Create a temp Go program with limited tokens
cat > /tmp/ask_granite_temp.go << 'EOFILE'
package main

import (
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
)

func main() {
	modelDir := "./models/granite-350m"

	config := nanovllm.NewConfig(
		modelDir,
		nanovllm.WithMaxNumSeqs(1),
		nanovllm.WithMaxModelLen(512),
		nanovllm.WithMaxNumBatchedTokens(512),
	)

	llm := nanovllm.NewLLM(config)
	defer llm.Close()

	// Only 20 tokens for reasonable speed
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(20),
	)

	prompts := os.Args[1:]
	outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("GRANITE RESPONSE (20 tokens)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	for _, output := range outputs {
		fmt.Printf("\nQ: %s\n", prompts[0])
		fmt.Printf("A: %s\n", output.Text)
		fmt.Printf("\n(Generated %d tokens)\n", len(output.TokenIDs))
	}
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}
EOFILE

# Build and run
go build -o /tmp/ask_granite /tmp/ask_granite_temp.go 2>/dev/null || {
    echo "Build failed. Make sure you're in the nano-vllm-go directory."
    exit 1
}

MODEL_DIR=./models/granite-350m /tmp/ask_granite "$1"

# Cleanup
rm -f /tmp/ask_granite /tmp/ask_granite_temp.go
