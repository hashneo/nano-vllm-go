package main

import (
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Nano-vLLM-Go - HTTP Backend Test")
	fmt.Println("==================================")
	fmt.Println()

	// Server URL
	serverURL := os.Getenv("MODEL_SERVER")
	if serverURL == "" {
		serverURL = "http://localhost:8000"
	}

	fmt.Printf("Connecting to server: %s\n", serverURL)

	// Create HTTP model runner
	modelRunner, err := purego.NewHTTPModelRunner(serverURL)
	if err != nil {
		log.Fatalf("Failed to connect to server: %v\n", err)
		fmt.Println("\nMake sure the Python server is running:")
		fmt.Println("  python3 server.py")
		return
	}
	defer modelRunner.Close()

	// Create HTTP tokenizer
	tokenizer := purego.NewHTTPTokenizer(serverURL, 151643)

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(32),
		nanovllm.WithMaxNumBatchedTokens(2048),
		nanovllm.WithEOS(tokenizer.EOSTokenID()),
	)

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(150),
	)

	// Get questions
	questions := []string{
		"What is the capital of France?",
		"Explain quantum computing in simple terms.",
		"What is 2 + 2?",
	}

	// Use command line arguments if provided
	if len(os.Args) > 1 {
		questions = os.Args[1:]
	}

	fmt.Printf("\nProcessing %d question(s)...\n\n", len(questions))

	// Generate
	outputs, err := llm.GenerateSimple(questions, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print results
	separator := "═══════════════════════════════════════════════════════════════════"
	fmt.Println("\n" + separator)
	fmt.Println("RESULTS")
	fmt.Println(separator)

	for i, output := range outputs {
		fmt.Printf("\n📝 Question %d: %s\n", i+1, questions[i])
		fmt.Printf("💬 Answer: %s\n", output.Text)
		fmt.Printf("📊 Tokens: %d\n", len(output.TokenIDs))
	}

	fmt.Println("\n" + separator)
	fmt.Println("✓ Test complete!")
	fmt.Println()
}
