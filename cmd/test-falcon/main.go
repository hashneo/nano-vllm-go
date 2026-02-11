package main

import (
	"fmt"
	"log"
	"time"

	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)

func main() {
	modelDir := "./models/falcon-7b-instruct"

	fmt.Println("Loading Falcon-7B-Instruct...")
	start := time.Now()

	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	tokenizer, err := purego.NewUniversalTokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	fmt.Printf("✓ Model loaded in %v\n\n", time.Since(start))

	// Falcon prompt format
	prompt := "User: What is the capital of Germany?\nAssistant:"
	fmt.Printf("Prompt: %s\n\n", prompt)

	// Encode prompt
	promptTokens, err := tokenizer.Encode(prompt)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}
	fmt.Printf("Encoded to %d tokens\n\n", len(promptTokens))

	// Generate 30 tokens with KV caching
	fmt.Println("Generating response with KV caching...")

	var kvCache *tensor.KVCache = nil
	allTokens := promptTokens

	// Prefill - process all prompt tokens
	fmt.Printf("Prefill: processing %d tokens...\n", len(promptTokens))
	prefillStart := time.Now()
	logits, newCache := model.ForwardWithCache(promptTokens, nil, 0)
	kvCache = newCache
	prefillTime := time.Since(prefillStart)
	fmt.Printf("Prefill done in %v (%.2f tok/s)\n\n", prefillTime, float64(len(promptTokens))/prefillTime.Seconds())

	// Get first token
	lastLogits := model.GetLogitsForLastToken(logits)
	nextToken := argmax(lastLogits)
	allTokens = append(allTokens, nextToken)

	decoded, _ := tokenizer.Decode([]int{nextToken})
	fmt.Printf("Generated: %s", decoded)

	// Decode - generate remaining tokens one at a time
	decodeStart := time.Now()
	for i := 0; i < 29; i++ {
		// Process only the new token with cache
		inputTokens := []int{allTokens[len(allTokens)-1]}
		posOffset := len(allTokens) - 1

		logits, newCache = model.ForwardWithCache(inputTokens, kvCache, posOffset)
		kvCache = newCache

		lastLogits = model.GetLogitsForLastToken(logits)
		nextToken = argmax(lastLogits)

		allTokens = append(allTokens, nextToken)

		decoded, _ = tokenizer.Decode([]int{nextToken})
		fmt.Printf("%s", decoded)

		if nextToken == tokenizer.EOSTokenID() {
			break
		}
	}
	decodeTime := time.Since(decodeStart)

	fullText, _ := tokenizer.Decode(allTokens[len(promptTokens):])
	fmt.Printf("\n\n=== Full Response ===\n%s\n", fullText)

	fmt.Printf("\nStatistics:\n")
	fmt.Printf("  Prefill: %d tokens in %v (%.2f tok/s)\n",
		len(promptTokens), prefillTime, float64(len(promptTokens))/prefillTime.Seconds())
	fmt.Printf("  Decode: %d tokens in %v (%.2f tok/s)\n",
		len(allTokens)-len(promptTokens), decodeTime, float64(len(allTokens)-len(promptTokens))/decodeTime.Seconds())
}

func argmax(logits []float32) int {
	maxIdx := 0
	maxVal := logits[0]
	for j := 1; j < len(logits); j++ {
		if logits[j] > maxVal {
			maxVal = logits[j]
			maxIdx = j
		}
	}
	return maxIdx
}
