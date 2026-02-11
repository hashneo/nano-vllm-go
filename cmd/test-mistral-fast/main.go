package main

import (
	"fmt"
	"log"

	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)

func main() {
	modelDir := "./models/mistral-7b-instruct"

	fmt.Println("Loading Mistral-7B-Instruct...")
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	tokenizer, err := purego.NewUniversalTokenizer(modelDir)
	if err != nil {
		log.Fatalf("Failed to load tokenizer: %v", err)
	}

	// Mistral uses <s>[INST] prompt [/INST] format
	prompt := "<s>[INST] What is the capital of Germany? [/INST]"
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
	logits, newCache := model.ForwardWithCache(promptTokens, nil, 0)
	kvCache = newCache

	// Get first token
	lastLogits := model.GetLogitsForLastToken(logits)
	nextToken := argmax(lastLogits)
	allTokens = append(allTokens, nextToken)

	decoded, _ := tokenizer.Decode([]int{nextToken})
	fmt.Printf("%s", decoded)

	// Print top 5 after prefill
	fmt.Printf("\n[DEBUG] Top 5 after prefill: ")
	printTop5(lastLogits, 5)

	// Decode - generate remaining tokens one at a time
	for i := 0; i < 29; i++ {
		// Process only the new token with cache
		inputTokens := []int{allTokens[len(allTokens)-1]}
		posOffset := len(allTokens) - 1

		// Debug: Check embeddings for first few tokens
		if i < 3 {
			fmt.Printf("\n[DEBUG step %d] Input token: %d\n", i+1, inputTokens[0])
		}

		logits, newCache = model.ForwardWithCache(inputTokens, kvCache, posOffset)
		kvCache = newCache

		// Debug cache growth
		if i < 3 {
			k, _ := kvCache.GetLayer(0)
			if k != nil {
				fmt.Printf("[DEBUG] Cache size at step %d: %v\n", i+1, k.Shape)
			}
		}

		lastLogits = model.GetLogitsForLastToken(logits)
		nextToken = argmax(lastLogits)

		allTokens = append(allTokens, nextToken)

		decoded, _ = tokenizer.Decode([]int{nextToken})
		fmt.Printf("%s", decoded)

		// Debug first few tokens
		if i < 3 {
			fmt.Printf("\n[DEBUG step %d] Token %d, top 5: ", i+1, nextToken)
			printTop5(lastLogits, 5)
			fmt.Println()
		}

		if nextToken == tokenizer.EOSTokenID() {
			break
		}
	}

	fullText, _ := tokenizer.Decode(allTokens[len(promptTokens):])
	fmt.Printf("\n\n=== Full Response ===\n%s\n", fullText)
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

func printTop5(logits []float32, count int) {
	type tl struct {
		id    int
		logit float32
	}
	top := make([]tl, 0)
	for i, l := range logits {
		if len(top) < count {
			top = append(top, tl{i, l})
			for j := len(top) - 1; j > 0 && top[j].logit > top[j-1].logit; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		} else if l > top[count-1].logit {
			top[count-1] = tl{i, l}
			for j := count - 1; j > 0 && top[j].logit > top[j-1].logit; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}
	for _, t := range top {
		fmt.Printf("%d(%.1f) ", t.id, t.logit)
	}
}
