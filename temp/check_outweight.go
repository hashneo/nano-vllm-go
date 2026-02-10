package main
import ("fmt"; "nano-vllm-go/purego/tensor")
func main() {
	model, _ := tensor.LoadModelFromDirectory("./models/gpt2-small")
	mha := model.Blocks[0].Attention.(*tensor.MultiHeadAttention)
	fmt.Printf("OutWeight shape: %v\n", mha.OutWeight.Shape)
	fmt.Printf("OutWeight [0:3, 0:3]:\n")
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%.4f ", mha.OutWeight.Data[i*768+j])
		}
		fmt.Println()
	}
}
