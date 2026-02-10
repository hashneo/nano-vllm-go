package main

import (
	"fmt"
	"log"

	"nano-vllm-go/purego/tensor"
)

func main() {
	fmt.Println("Verifying GPT-2 Weights")
	fmt.Println("=======================\n")

	// Load model
	model, err := tensor.LoadModelFromDirectory("./models/gpt2-small")
	if err != nil {
		log.Fatalf("Failed to load: %v", err)
	}

	// Check token embedding for Paris (token 6342)
	hidden := model.Config.Hidden
	token6342 := 6342

	fmt.Printf("Token %d (should be 'Paris'):\n", token6342)
	fmt.Printf("First 10 embedding values: ")
	for i := 0; i < 10; i++ {
		fmt.Printf("%.4f ", model.TokenEmbedding.Data[token6342*hidden+i])
	}
	fmt.Println()

	// Expected values from Python
	expected := []float32{-0.2195, 0.0422, 0.1208, -0.2595, 0.0278, -0.0512, -0.2290, -0.0217, -0.0412, 0.2218}
	fmt.Printf("Expected values:           ")
	for _, v := range expected {
		fmt.Printf("%.4f ", v)
	}
	fmt.Println()

	// Check if they match
	match := true
	for i := 0; i < 10; i++ {
		actual := model.TokenEmbedding.Data[token6342*hidden+i]
		diff := actual - expected[i]
		if diff < -0.01 || diff > 0.01 {
			match = false
		}
	}

	if match {
		fmt.Println("\n✓ Weights match! Model loaded correctly.")
	} else {
		fmt.Println("\n❌ Weights don't match! Model loading issue.")
	}

	// Check first layer QKV weights
	fmt.Println("\nChecking first attention layer:")
	if model.Blocks[0].Attention != nil {
		if mha, ok := model.Blocks[0].Attention.(*tensor.MultiHeadAttention); ok {
			if mha.QWeight != nil {
				fmt.Printf("Q weight shape: %v\n", mha.QWeight.Shape)
				fmt.Printf("Q weight first 5 values: ")
				for i := 0; i < 5; i++ {
					fmt.Printf("%.4f ", mha.QWeight.Data[i])
				}
				fmt.Println()
			}
		}
	}
}
