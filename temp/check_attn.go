package main
import (
	"fmt"
	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)
func main() {
	model, _ := tensor.LoadModelFromDirectory("./models/gpt2-small")
	tokenizer, _ := purego.NewBPETokenizer("./models/gpt2-small")
	tokenIDs, _ := tokenizer.Encode("The capital of France is")
	x := tensor.NewTensor(1, len(tokenIDs), 768)
	for i, tokenID := range tokenIDs {
		for j := 0; j < 768; j++ {
			x.Data[i*768+j] = model.TokenEmbedding.Data[tokenID*768+j] + model.PosEmbedding.Data[i*768+j]
		}
	}
	ln1Out := model.Blocks[0].AttnLN.Forward(x)
	fmt.Printf("After LN1 [0,0,0:5]: ")
	for i := 0; i < 5; i++ {
		fmt.Printf("0.0000 ", ln1Out.Data[i])
	}
	fmt.Println()
	mha := model.Blocks[0].Attention.(*tensor.MultiHeadAttention)
	attnOut, _, _ := mha.ForwardWithCache(ln1Out, nil, nil)
	fmt.Printf("After attention [0,0,0:5]: ")
	for i := 0; i < 5; i++ {
		fmt.Printf("0.0000 ", attnOut.Data[i])
	}
	fmt.Println()
}
