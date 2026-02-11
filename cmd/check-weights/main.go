package main

import (
	"fmt"
	"log"

	"nano-vllm-go/purego/tensor"
)

func main() {
	// Load model
	modelPath := "./models/llama-3.2-1b-instruct"
	model, err := tensor.LoadModelFromDirectory(modelPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	fmt.Println("=== Llama 3.2 1B Weight Analysis ===")

	// Check token embeddings
	fmt.Printf("\nToken Embedding: %v\n", model.TokenEmbedding.Shape)
	fmt.Printf("  First 10 values: ")
	for i := 0; i < 10; i++ {
		fmt.Printf("%.6f ", model.TokenEmbedding.Data[i])
	}
	fmt.Println()
	fmt.Printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n",
		minFloat32(model.TokenEmbedding.Data), maxFloat32(model.TokenEmbedding.Data), meanFloat32(model.TokenEmbedding.Data))

	// Check first layer attention weights
	if len(model.Blocks) > 0 {
		fmt.Println("\n=== Layer 0 Attention ===")
		block := model.Blocks[0]

		// Try to cast to GQA
		if gqa, ok := block.Attention.(*tensor.GroupedQueryAttention); ok {
			fmt.Printf("QWeight shape: %v\n", gqa.QWeight.Shape)
			fmt.Printf("  First 10 values: ")
			for i := 0; i < 10; i++ {
				fmt.Printf("%.6f ", gqa.QWeight.Data[i])
			}
			fmt.Println()
			fmt.Printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n",
				minFloat32(gqa.QWeight.Data), maxFloat32(gqa.QWeight.Data), meanFloat32(gqa.QWeight.Data))

			fmt.Printf("\nKWeight shape: %v\n", gqa.KWeight.Shape)
			fmt.Printf("  First 10 values: ")
			for i := 0; i < 10; i++ {
				fmt.Printf("%.6f ", gqa.KWeight.Data[i])
			}
			fmt.Println()
			fmt.Printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n",
				minFloat32(gqa.KWeight.Data), maxFloat32(gqa.KWeight.Data), meanFloat32(gqa.KWeight.Data))

			fmt.Printf("\nVWeight shape: %v\n", gqa.VWeight.Shape)
			fmt.Printf("  First 10 values: ")
			for i := 0; i < 10; i++ {
				fmt.Printf("%.6f ", gqa.VWeight.Data[i])
			}
			fmt.Println()
			fmt.Printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n",
				minFloat32(gqa.VWeight.Data), maxFloat32(gqa.VWeight.Data), meanFloat32(gqa.VWeight.Data))

			fmt.Printf("\nOutWeight shape: %v\n", gqa.OutWeight.Shape)
			fmt.Printf("  First 10 values: ")
			for i := 0; i < 10; i++ {
				fmt.Printf("%.6f ", gqa.OutWeight.Data[i])
			}
			fmt.Println()
			fmt.Printf("  Stats: min=%.6f, max=%.6f, mean=%.6f\n",
				minFloat32(gqa.OutWeight.Data), maxFloat32(gqa.OutWeight.Data), meanFloat32(gqa.OutWeight.Data))

			// Check if shapes are consistent
			fmt.Println("\n=== Shape Verification ===")
			hidden := gqa.Hidden
			numHeads := gqa.NumHeads
			numKVHeads := gqa.NumKVHeads
			headDim := gqa.HeadDim

			fmt.Printf("hidden=%d, num_heads=%d, num_kv_heads=%d, head_dim=%d\n",
				hidden, numHeads, numKVHeads, headDim)
			fmt.Printf("Expected: num_heads * head_dim = %d (should be %d)\n",
				numHeads*headDim, hidden)
			fmt.Printf("Expected: num_kv_heads * head_dim = %d\n", numKVHeads*headDim)

			// Verify weight shapes match expectations
			expectedQShape := []int{hidden, hidden}
			expectedKVShape := []int{hidden, numKVHeads * headDim}

			fmt.Printf("\nQWeight: expected %v, actual %v, match: %v\n",
				expectedQShape, gqa.QWeight.Shape, shapeMatch(expectedQShape, gqa.QWeight.Shape))
			fmt.Printf("KWeight: expected %v, actual %v, match: %v\n",
				expectedKVShape, gqa.KWeight.Shape, shapeMatch(expectedKVShape, gqa.KWeight.Shape))
			fmt.Printf("VWeight: expected %v, actual %v, match: %v\n",
				expectedKVShape, gqa.VWeight.Shape, shapeMatch(expectedKVShape, gqa.VWeight.Shape))

			// Verify transpose by checking first column
			fmt.Println("\n=== Transpose Verification ===")
			fmt.Println("PyTorch Q weight[0, 0:10] (first row):")
			fmt.Println("  [-0.0179, 0.0066, 0.0247, -0.0048, -0.0139, -0.0369, -0.0037, -0.0129, 0.0044, 0.0211]")
			fmt.Println("\nOur QWeight[:, 0] (first column, after transpose):")
			fmt.Printf("  [")
			for i := 0; i < 10; i++ {
				idx := i*gqa.QWeight.Shape[1] + 0  // row i, column 0
				fmt.Printf("%.4f", gqa.QWeight.Data[idx])
				if i < 9 {
					fmt.Printf(", ")
				}
			}
			fmt.Println("]")
			fmt.Println("\nThese should match! PyTorch's first row becomes our first column after transpose.")
		}
	}
}

func minFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	min := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
	}
	return min
}

func maxFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	max := data[0]
	for _, v := range data {
		if v > max {
			max = v
		}
	}
	return max
}

func meanFloat32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	sum := float32(0)
	for _, v := range data {
		sum += v
	}
	return sum / float32(len(data))
}

func shapeMatch(expected, actual []int) bool {
	if len(expected) != len(actual) {
		return false
	}
	for i := range expected {
		if expected[i] != actual[i] {
			return false
		}
	}
	return true
}
