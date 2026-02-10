package main
import ("fmt"; "nano-vllm-go/purego/tensor")
func main() {
	model, _ := tensor.LoadModelFromDirectory("./models/gpt2-small")
	mha := model.Blocks[0].Attention.(*tensor.MultiHeadAttention)
	fmt.Printf("Q[0, 0:3]: ")
	for i := 0; i < 3; i++ {
		fmt.Printf("%.4f ", mha.QWeight.Data[i])
	}
	fmt.Println()
}
