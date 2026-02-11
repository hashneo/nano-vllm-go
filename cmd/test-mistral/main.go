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
	tokens, err := tokenizer.Encode(prompt)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}
	fmt.Printf("Encoded to %d tokens\n\n", len(tokens))

	// Generate 30 tokens with greedy decoding
	fmt.Println("Generating response...")
	allTokens := tokens

	for i := 0; i < 30; i++ {
		logits := model.Forward(allTokens)
		lastLogits := model.GetLogitsForLastToken(logits)

		// Argmax
		maxIdx := 0
		maxVal := lastLogits[0]
		for j := 1; j < len(lastLogits); j++ {
			if lastLogits[j] > maxVal {
				maxVal = lastLogits[j]
				maxIdx = j
			}
		}

		allTokens = append(allTokens, maxIdx)

		// Decode and print
		decoded, _ := tokenizer.Decode([]int{maxIdx})
		fmt.Printf("%s", decoded)

		if maxIdx == tokenizer.EOSTokenID() {
			break
		}
	}

	fullText, _ := tokenizer.Decode(allTokens[len(tokens):])
	fmt.Printf("\n\n=== Full Response ===\n%s\n", fullText)
}
