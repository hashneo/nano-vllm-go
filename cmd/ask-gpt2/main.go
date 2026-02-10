package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"nano-vllm-go/nanovllm"
	"nano-vllm-go/purego"
)

func main() {
	// Define command-line flags
	creative := flag.Bool("creative", false, "Use creative mode with higher temperature for storytelling")
	temperature := flag.Float64("temp", 0.0, "Temperature for sampling (overrides mode defaults)")
	maxTokens := flag.Int("max-tokens", 0, "Maximum tokens to generate (overrides mode defaults)")
	repPenalty := flag.Float64("rep-penalty", 0.0, "Repetition penalty (1.0=no penalty, >1.0=penalize repeats)")

	flag.Parse()

	modelDir := "./models/gpt2-small"

	// Get question from remaining args after flags
	question := "The capital of France is"
	if flag.NArg() > 0 {
		question = strings.Join(flag.Args(), " ")
	}

	// Determine sampling parameters based on mode
	var temp float64
	var topP float32
	var topK int
	var maxTok int
	var repPen float32

	if *creative {
		// Creative mode: higher temperature, nucleus sampling, longer output
		temp = 0.8
		topP = 0.95
		topK = 50
		maxTok = 100
		repPen = 1.15 // Moderate penalty for creative text
	} else {
		// Factual mode: very low temperature for deterministic, precise answers
		temp = 0.3  // Low but not too low to avoid complete determinism
		topP = 1.0  // Disabled for factual mode
		topK = 5    // Only consider top 5 most likely tokens
		maxTok = 20
		repPen = 1.5 // Strong penalty to prevent loops
	}

	// Allow manual overrides
	if *temperature > 0 {
		temp = *temperature
	}
	if *maxTokens > 0 {
		maxTok = *maxTokens
	}
	if *repPenalty > 0 {
		repPen = float32(*repPenalty)
	}

	mode := "Factual"
	if *creative {
		mode = "Creative"
	}

	fmt.Printf("GPT-2 Question Answering (%s Mode)\n", mode)
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("Question: %s\n", question)
	fmt.Printf("Temperature: %.2f | Top-P: %.2f | Top-K: %d | Rep-Penalty: %.2f | Max Tokens: %d\n\n", temp, topP, topK, repPen, maxTok)

	// Create config
	config := nanovllm.NewConfig(
		modelDir,
		nanovllm.WithMaxNumSeqs(1),
		nanovllm.WithMaxModelLen(512),
		nanovllm.WithMaxNumBatchedTokens(512),
	)

	// Create model runner
	modelRunner, err := nanovllm.NewTensorModelRunner(modelDir)
	if err != nil {
		log.Fatalf("Failed to create model runner: %v\n", err)
	}

	// Set sampling parameters (temperature, top-p, top-k, repetition penalty)
	modelRunner.SetSamplingParamsWithRepetition(float32(temp), topP, topK, repPen)

	// Create proper BPE tokenizer
	tokenizer, err := purego.NewBPETokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v\n", err)
	}

	// Create LLM with real model
	llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
	defer llm.Close()

	// Set up sampling parameters
	// With KV caching, generation is now O(N) - much faster!
	samplingParams := nanovllm.NewSamplingParams(
		nanovllm.WithTemperature(temp),
		nanovllm.WithMaxTokens(maxTok),
	)

	// Generate
	fmt.Println("Generating response...")
	outputs, err := llm.GenerateSimple([]string{question}, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v\n", err)
	}

	// Print result
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("Full text: %s%s\n", question, outputs[0].Text)
	fmt.Printf("Answer only: %s\n", outputs[0].Text)
	fmt.Printf("\nGenerated %d tokens\n", len(outputs[0].TokenIDs))
	fmt.Printf("Token IDs: %v\n", outputs[0].TokenIDs)

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
