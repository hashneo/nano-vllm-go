package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run ./cmd/ask-llama/main.go \"Your question here\"")
		fmt.Println("\nExample:")
		fmt.Println("  go run ./cmd/ask-llama/main.go \"What is the capital of France?\"")
		os.Exit(1)
	}

	question := os.Args[1]
	modelDir := "./models/llama-3.2-1b-instruct"

	fmt.Println("Loading Llama-3.2-1B-Instruct...")
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	tokenizer, err := purego.NewUniversalTokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Format as Llama 3 chat
	prompt := fmt.Sprintf("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", question)

	// Encode
	promptTokens, err := tokenizer.Encode(prompt)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}

	fmt.Printf("\nQuestion: %s\n\n", question)

	// Initialize KV cache
	kvCache := tensor.NewKVCache(model.Config.NumLayers)

	// Prefill
	start := time.Now()
	logits, kvCache := model.ForwardWithCache(promptTokens, kvCache, 0)
	prefillTime := time.Since(start)

	lastLogits := model.GetLogitsForLastToken(logits)
	nextToken := argmax(lastLogits)

	allTokens := append(promptTokens, nextToken)
	decoded, _ := tokenizer.Decode([]int{nextToken})
	fmt.Printf("Answer: %s", decoded)

	// Generate up to 100 tokens
	decodeStart := time.Now()
	tokensGenerated := 1
	for i := 0; i < 99; i++ {
		logits, kvCache = model.ForwardWithCache([]int{allTokens[len(allTokens)-1]}, kvCache, len(allTokens)-1)
		lastLogits = model.GetLogitsForLastToken(logits)
		nextToken = argmax(lastLogits)

		decoded, _ = tokenizer.Decode([]int{nextToken})
		fmt.Printf("%s", decoded)

		allTokens = append(allTokens, nextToken)
		tokensGenerated++

		if nextToken == tokenizer.EOSTokenID() {
			break
		}
	}
	decodeTime := time.Since(decodeStart)

	fmt.Printf("\n\n")
	fmt.Printf("Stats: Prefill %.2f tok/s, Decode %.2f tok/s\n",
		float64(len(promptTokens))/prefillTime.Seconds(),
		float64(tokensGenerated)/decodeTime.Seconds())
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
