package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/your-username/nano-vllm-go/nanovllm"
)

func main() {
	fmt.Println("Nano-vLLM-Go Benchmark")
	fmt.Println("======================")
	fmt.Println()

	// Configuration
	numRequests := 256
	minInputLen := 100
	maxInputLen := 1024
	minOutputLen := 100
	maxOutputLen := 1024

	fmt.Printf("Configuration:\n")
	fmt.Printf("  Number of requests: %d\n", numRequests)
	fmt.Printf("  Input length: %d-%d tokens\n", minInputLen, maxInputLen)
	fmt.Printf("  Output length: %d-%d tokens\n", minOutputLen, maxOutputLen)
	fmt.Println()

	// Create config
	config := nanovllm.NewConfig(
		".",
		nanovllm.WithMaxNumSeqs(512),
		nanovllm.WithMaxNumBatchedTokens(16384),
		nanovllm.WithEnforceEager(true),
		nanovllm.WithTensorParallelSize(1),
	)

	// Create LLM
	llm := nanovllm.NewLLM(config)
	defer llm.Close()

	// Generate random prompts
	prompts := make([][]int, numRequests)
	samplingParams := make([]*nanovllm.SamplingParams, numRequests)
	totalExpectedTokens := 0

	rand.Seed(time.Now().UnixNano())
	for i := 0; i < numRequests; i++ {
		inputLen := minInputLen + rand.Intn(maxInputLen-minInputLen+1)
		outputLen := minOutputLen + rand.Intn(maxOutputLen-minOutputLen+1)

		// Generate random token IDs
		tokens := make([]int, inputLen)
		for j := 0; j < inputLen; j++ {
			tokens[j] = rand.Intn(32000)
		}
		prompts[i] = tokens

		samplingParams[i] = nanovllm.NewSamplingParams(
			nanovllm.WithTemperature(0.6),
			nanovllm.WithMaxTokens(outputLen),
		)

		totalExpectedTokens += outputLen
	}

	fmt.Println("Starting benchmark...")
	fmt.Println()

	// Convert prompts to interface{}
	promptsInterface := make([]interface{}, numRequests)
	for i, p := range prompts {
		promptsInterface[i] = p
	}

	// Run benchmark
	startTime := time.Now()
	outputs, err := llm.Generate(promptsInterface, samplingParams, true)
	if err != nil {
		log.Fatalf("Generation failed: %v", err)
	}
	elapsed := time.Since(startTime).Seconds()

	// Calculate statistics
	totalOutputTokens := 0
	for _, output := range outputs {
		totalOutputTokens += len(output.TokenIDs)
	}

	throughput := float64(totalOutputTokens) / elapsed

	fmt.Println()
	fmt.Println("Benchmark Results:")
	fmt.Println("==================")
	fmt.Printf("Total requests: %d\n", numRequests)
	fmt.Printf("Total output tokens: %d\n", totalOutputTokens)
	fmt.Printf("Time elapsed: %.2f seconds\n", elapsed)
	fmt.Printf("Throughput: %.2f tokens/sec\n", throughput)
	fmt.Printf("Average latency: %.2f ms/request\n", elapsed*1000/float64(numRequests))
	fmt.Println()

	// Note about mock implementation
	fmt.Println("Note: This benchmark uses a mock model runner.")
	fmt.Println("For real performance measurements, integrate with an actual inference backend.")
}
