package main

import (
	"fmt"
	"log"
	"time"

	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)

func main() {
	modelDir := "./models/llama-3.2-1b-instruct"

	fmt.Println("Loading Llama-3.2-1B-Instruct...")
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

	// Use correct tokens from Python tokenizer
	// Prompt: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of Germany?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	promptTokens := []int{128000, 128006, 882, 128007, 271, 3923, 374, 279, 6864, 315, 10057, 30, 128009, 128006, 78191, 128007, 271}

	fmt.Printf("Using pre-tokenized prompt (%d tokens)\n", len(promptTokens))
	fmt.Println("Question: What is the capital of Germany?\n")

	// Initialize KV cache
	kvCache := tensor.NewKVCache(model.Config.NumLayers)
	allTokens := make([]int, len(promptTokens))
	copy(allTokens, promptTokens)

	// Prefill phase
	fmt.Println("Generating response...")
	prefillStart := time.Now()
	logits, kvCache := model.ForwardWithCache(allTokens, kvCache, 0)
	prefillTime := time.Since(prefillStart)

	lastLogits := model.GetLogitsForLastToken(logits)
	nextToken := argmax(lastLogits)

	decoded, _ := tokenizer.Decode([]int{nextToken})
	fmt.Printf("First token: '%s' (ID: %d)\n", decoded, nextToken)
	fmt.Printf("Prefill: %v (%.2f tok/s)\n\n", prefillTime, float64(len(allTokens))/prefillTime.Seconds())

	allTokens = append(allTokens, nextToken)
	fmt.Printf("Output: %s", decoded)

	// Generate up to 50 tokens
	for i := 0; i < 50; i++ {
		decodeStart := time.Now()
		logits, kvCache = model.ForwardWithCache([]int{allTokens[len(allTokens)-1]}, kvCache, len(allTokens)-1)
		lastLogits = model.GetLogitsForLastToken(logits)
		nextToken = argmax(lastLogits)
		decodeTime := time.Since(decodeStart)

		decoded, _ = tokenizer.Decode([]int{nextToken})
		fmt.Printf("%s", decoded)

		allTokens = append(allTokens, nextToken)

		if i == 0 {
			fmt.Printf(" (decode: %.1f tok/s)", 1.0/decodeTime.Seconds())
		}

		if nextToken == tokenizer.EOSTokenID() {
			fmt.Println("\n\n✓ Generation complete (hit EOS)")
			break
		}
	}

	fmt.Println()
}

func argmax(data []float32) int {
	if len(data) == 0 {
		return 0
	}
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}
