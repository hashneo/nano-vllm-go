package main

import (
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Nano-vLLM-Go - Pure Go Transformer")
	fmt.Println("===================================")
	fmt.Println()

	// Get model path
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "./models/gpt2/model.safetensors"
	}

	tokenizerPath := os.Getenv("TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "./models/gpt2"
	}

	fmt.Printf("Loading model from: %s\n", modelPath)
	fmt.Printf("Loading tokenizer from: %s\n", tokenizerPath)
	fmt.Println()

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(8),
		nanovllm.WithMaxNumBatchedTokens(512),
		nanovllm.WithEOS(50256), // GPT-2 EOS
	)

	// Create model runner
	fmt.Println("Loading pure Go transformer...")
	modelRunner, err := purego.NewNativeModelRunner(modelPath, config)
	if err != nil {
		log.Fatalf("Failed to create model runner: %v\n", err)
	}
	defer modelRunner.Close()

	fmt.Println()

	// Create tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := purego.NewGPT2Tokenizer(tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v\n", err)
	}

	fmt.Println()

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.8),
		nanovllm.WithMaxTokens(50),
	)

	// Test prompts
	prompts := []string{
		"Once upon a time",
		"The meaning of life is",
		"In the beginning",
	}

	// Allow custom prompt from command line
	if len(os.Args) > 1 {
		prompts = os.Args[1:]
	}

	fmt.Printf("Generating text for %d prompt(s)...\n", len(prompts))
	fmt.Println()

	// Generate
	outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print results
	separator := "═══════════════════════════════════════════════════════════════════"
	fmt.Println("\n" + separator)
	fmt.Println("RESULTS")
	fmt.Println(separator)

	for i, output := range outputs {
		fmt.Printf("\n📝 Prompt %d: %s\n", i+1, prompts[i])
		fmt.Printf("💬 Generated: %s\n", output.Text)
		fmt.Printf("📊 Tokens: %d\n", len(output.TokenIDs))
	}

	fmt.Println("\n" + separator)
	fmt.Println("✓ Generation complete!")
	fmt.Println()
	fmt.Println("Note: This is a PURE GO transformer implementation!")
	fmt.Println("No Python, no ONNX, no external libraries - just Go!")
}
