package main

import (
	"fmt"
	"log"
	"os"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	fmt.Println("Nano-vLLM-Go - Universal Transformer Runner")
	fmt.Println("============================================")
	fmt.Println()

	// Get model directory
	modelDir := os.Getenv("MODEL_DIR")
	if modelDir == "" {
		// Check for different model directories
		possibleDirs := []string{
			"./models/granite-350m",
			"./models/granite-1b",
			"./models/falcon-7b",
			"./models/gpt2",
			"./models/llama-7b",
			"./models/qwen2",
		}

		for _, dir := range possibleDirs {
			if _, err := os.Stat(dir); err == nil {
				modelDir = dir
				break
			}
		}

		if modelDir == "" {
			fmt.Println("No model found. Set MODEL_DIR or download a model:")
			fmt.Println()
			fmt.Println("Granite 4 Nano 350M (small, hybrid architecture with Mamba2):")
			fmt.Println("  ./scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m")
			fmt.Println()
			fmt.Println("Granite 4 Nano 1B (medium, hybrid architecture with Mamba2):")
			fmt.Println("  ./scripts/download_model.py --model ibm-granite/granite-4.0-h-1b --output ./models/granite-1b --fp16")
			fmt.Println()
			fmt.Println("GPT-2 (124M, fast to download):")
			fmt.Println("  ./scripts/download_model.py --model gpt2 --output ./models/gpt2")
			fmt.Println()
			fmt.Println("Falcon 7B (7B, production quality):")
			fmt.Println("  ./scripts/download_model.py --model tiiuae/falcon-7b --output ./models/falcon-7b --fp16")
			fmt.Println()
			fmt.Println("Llama 2 7B (7B, state of the art):")
			fmt.Println("  ./scripts/download_model.py --model meta-llama/Llama-2-7b-hf --output ./models/llama-7b --fp16")
			fmt.Println()
			os.Exit(1)
		}
	}

	fmt.Printf("Model directory: %s\n", modelDir)
	fmt.Println()

	// Create engine config
	// For long-context models like Granite (32K), we limit the actual batch size
	// but set max_model_len to match the model's capability
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(16),
		nanovllm.WithMaxModelLen(8192),         // Support up to 8K tokens per sequence
		nanovllm.WithMaxNumBatchedTokens(8192), // Must be >= max_model_len
	)

	// Load model (auto-detects architecture!)
	fmt.Println("Loading model...")
	modelRunner, err := purego.NewGenericModelRunner(modelDir, config)
	if err != nil {
		log.Fatalf("Failed to load model: %v\n", err)
	}
	defer modelRunner.Close()
	fmt.Println()

	// Update config with model's EOS token
	modelConfig := modelRunner.GetModelConfig()
	config.EOS = modelConfig.EOSTokenID

	// Load tokenizer
	fmt.Println("Loading tokenizer...")
	tokenizer, err := purego.NewUniversalTokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v\n", err)
	}
	fmt.Println()

	// Create LLM engine
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(0.7),
		nanovllm.WithMaxTokens(100),
	)

	// Get prompts
	prompts := []string{
		"What is the capital of France?",
		"Explain artificial intelligence in one sentence.",
		"What is 15 * 23?",
	}

	// Use command line arguments if provided
	if len(os.Args) > 1 {
		prompts = os.Args[1:]
	}

	fmt.Printf("Generating responses for %d prompt(s)...\n", len(prompts))
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
		fmt.Printf("\n📝 Question %d: %s\n", i+1, prompts[i])
		fmt.Printf("💬 Answer: %s\n", output.Text)
		fmt.Printf("📊 Tokens: %d\n", len(output.TokenIDs))
	}

	fmt.Println("\n" + separator)
	fmt.Println("✓ Generation complete!")
	fmt.Println()

	// Show model info
	fmt.Printf("Model: %s (%s architecture)\n",
		modelConfig.ModelName,
		modelConfig.Architecture)
	fmt.Printf("  • Position encoding: %s\n", modelConfig.PositionType)
	fmt.Printf("  • Attention: %s (%d query heads, %d KV heads)\n",
		modelConfig.AttentionType,
		modelConfig.NumHeads,
		modelConfig.NumKVHeads)
	fmt.Printf("  • Block style: %s\n", modelConfig.BlockStyle)
	fmt.Println()
}
