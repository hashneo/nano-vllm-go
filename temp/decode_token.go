package main

import (
	"fmt"
	"log"
	"nano-vllm-go/purego"
)

func main() {
	tokenizer, err := purego.NewBPETokenizer("./models/gpt2-small")
	if err \!= nil {
		log.Fatal(err)
	}

	tokens := []int{357, 284, 262, 6342, 4881}
	for _, tok := range tokens {
		decoded, _ := tokenizer.Decode([]int{tok})
		fmt.Printf("Token %d: %q\n", tok, decoded)
	}
}
