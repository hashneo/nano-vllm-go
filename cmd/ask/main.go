package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"nano-vllm-go/purego"
	"nano-vllm-go/purego/tensor"
)

// ModelConfig defines configuration for each supported model
type ModelConfig struct {
	Name         string
	DefaultPath  string
	ChatFormat   string
	UsePythonTok bool
	Variants     map[string]string // For GPT-2 model size variants
}

var models = map[string]ModelConfig{
	"gpt2": {
		Name:         "GPT-2",
		DefaultPath:  "./models/gpt2-small",
		ChatFormat:   "completion",
		UsePythonTok: false,
		Variants: map[string]string{
			"small":  "./models/gpt2-small",
			"medium": "./models/gpt2-medium",
			"large":  "./models/gpt2-large",
			"xl":     "./models/gpt2-xl",
		},
	},
	"llama": {
		Name:         "Llama-3.2-1B-Instruct",
		DefaultPath:  "./models/llama-3.2-1b-instruct",
		ChatFormat:   "llama3",
		UsePythonTok: true,
	},
	"falcon": {
		Name:         "Falcon-7B-Instruct",
		DefaultPath:  "./models/falcon-7b-instruct",
		ChatFormat:   "falcon",
		UsePythonTok: true,
	},
	"granite": {
		Name:         "Granite-350M",
		DefaultPath:  "./models/granite-350m",
		ChatFormat:   "granite",
		UsePythonTok: true,
	},
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	modelType := os.Args[1]
	config, ok := models[modelType]
	if !ok {
		fmt.Fprintf(os.Stderr, "Error: Unknown model '%s'\n\n", modelType)
		printUsage()
		os.Exit(1)
	}

	// Parse flags for remaining arguments
	fs := flag.NewFlagSet("ask", flag.ExitOnError)

	// Set different defaults based on model type
	defaultTemp := 0.0  // Greedy for chat models (llama, falcon, granite)
	defaultMaxTokens := 100

	if modelType == "gpt2" {
		defaultTemp = 0.3      // Low temperature for more focused, less repetitive output
		defaultMaxTokens = 30  // Shorter outputs for completion models
	}

	temperature := fs.Float64("temp", defaultTemp, "Temperature for sampling (0=greedy)")
	maxTokens := fs.Int("max-tokens", defaultMaxTokens, "Maximum tokens to generate")
	modelVariant := fs.String("model", "small", "Model variant (GPT-2 only: small/medium/large/xl)")

	// Parse flags starting from os.Args[2:]
	fs.Parse(os.Args[2:])

	// Get question from remaining args
	if fs.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Error: Question required\n\n")
		printUsage()
		os.Exit(1)
	}
	question := strings.Join(fs.Args(), " ")

	// Determine model path
	modelDir := config.DefaultPath
	if modelType == "gpt2" {
		if variantPath, ok := config.Variants[*modelVariant]; ok {
			modelDir = variantPath
		} else {
			log.Fatalf("Invalid GPT-2 model variant: %s. Choose from: small, medium, large, xl\n", *modelVariant)
		}
	}

	// Check if model directory exists
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Error: Model not found at %s\n\n", modelDir)
		printDownloadInstructions(modelType, *modelVariant)
		os.Exit(1)
	}

	// Load model
	fmt.Printf("Loading %s...\n", config.Name)
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Load tokenizer
	var tokenizer *purego.UniversalTokenizer
	if config.UsePythonTok {
		tokenizer, err = purego.NewUniversalTokenizer(modelDir)
		if err != nil {
			log.Fatalf("Failed to load tokenizer: %v", err)
		}
	} else {
		// For GPT-2, we still need UniversalTokenizer for Decode/EOSTokenID
		tokenizer, err = purego.NewUniversalTokenizer(modelDir)
		if err != nil {
			log.Fatalf("Failed to load tokenizer: %v", err)
		}
	}

	// Format prompt with chat template
	prompt := formatPrompt(question, config.ChatFormat)

	// Encode prompt
	var promptTokens []int
	if config.UsePythonTok {
		promptTokens, err = encodeWithPython(modelDir, prompt)
		if err != nil {
			log.Fatalf("Failed to encode: %v", err)
		}
	} else {
		// Use Go tokenizer for GPT-2
		bpeTokenizer, err := purego.NewBPETokenizer(modelDir)
		if err != nil {
			log.Fatalf("Failed to create BPE tokenizer: %v", err)
		}
		promptTokens, err = bpeTokenizer.Encode(prompt)
		if err != nil {
			log.Fatalf("Failed to encode with BPE tokenizer: %v", err)
		}
	}

	fmt.Printf("\nQuestion: %s\n\n", question)

	// Generate response
	useTemperature := *temperature > 0
	generatedTokens, prefillTime, decodeTime := generateResponse(
		model, promptTokens, tokenizer, *maxTokens, *temperature, useTemperature,
	)

	// Print statistics
	fmt.Printf("\n\n")
	fmt.Printf("Stats: Prefill %.2f tok/s, Decode %.2f tok/s\n",
		float64(len(promptTokens))/prefillTime.Seconds(),
		float64(len(generatedTokens))/decodeTime.Seconds())
}

func printUsage() {
	fmt.Println("Usage: ask <model> [flags] \"question\"")
	fmt.Println("\nSupported models:")
	fmt.Println("  gpt2      - GPT-2 (small/medium/large/xl)")
	fmt.Println("  llama     - Llama-3.2-1B-Instruct")
	fmt.Println("  falcon    - Falcon-7B-Instruct")
	fmt.Println("  granite   - Granite-350M (experimental, may produce poor output)")
	fmt.Println("\nFlags:")
	fmt.Println("  -temp <float>      Temperature for sampling (default: 0.0 for greedy)")
	fmt.Println("  -max-tokens <int>  Maximum tokens to generate (default: 100)")
	fmt.Println("  -model <variant>   Model variant (GPT-2 only: small/medium/large/xl)")
	fmt.Println("\nExamples:")
	fmt.Println("  ask gpt2 \"The capital of France is\"          # Completion")
	fmt.Println("  ask llama \"What is the capital of France?\"   # Question")
	fmt.Println("  ask falcon \"What is the capital of Germany?\" # Question")
	fmt.Println("  ask gpt2 -model medium \"Once upon a time\"    # Larger model")
	fmt.Println("  ask llama -temp 0.7 -max-tokens 50 \"Tell me a story\"")
	fmt.Println()
	fmt.Println("Note: GPT-2 works with completion-style prompts.")
	fmt.Println("      Llama/Falcon work best with questions.")
}

func printDownloadInstructions(modelType, variant string) {
	fmt.Println("To download this model, run:")
	fmt.Println()

	switch modelType {
	case "gpt2":
		size := variant
		if size == "small" {
			fmt.Printf("  ./scripts/download_model.py --model gpt2 --output ./models/gpt2-%s\n", size)
		} else {
			fmt.Printf("  ./scripts/download_model.py --model gpt2-%s --output ./models/gpt2-%s\n", size, size)
		}
		fmt.Println()
		fmt.Println("Available sizes:")
		fmt.Println("  small  - 124M parameters (default)")
		fmt.Println("  medium - 355M parameters")
		fmt.Println("  large  - 774M parameters")
		fmt.Println("  xl     - 1.5B parameters")

	case "llama":
		fmt.Println("  ./scripts/download_model.py --model meta-llama/Llama-3.2-1B-Instruct --output ./models/llama-3.2-1b-instruct")
		fmt.Println()
		fmt.Println("NOTE: Llama models require accepting Meta's license agreement.")
		fmt.Println("      Visit https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
		fmt.Println("      and log in with: huggingface-cli login")
		fmt.Println("      Accept the license, then run the download command above.")

	case "falcon":
		fmt.Println("  ./scripts/download_model.py --model tiiuae/falcon-7b-instruct --output ./models/falcon-7b-instruct --fp16")
		fmt.Println()
		fmt.Println("NOTE: Falcon-7B is a 7 billion parameter model (~14GB download).")
		fmt.Println("      Using --fp16 to save memory and disk space.")

	case "granite":
		fmt.Println("  ./scripts/download_model.py --model ibm-granite/granite-3.0-1b-a400m-instruct --output ./models/granite-350m")
		fmt.Println()
		fmt.Println("NOTE: Downloads Granite 3.0 1B model (400M active parameters).")
	}

	fmt.Println()
	fmt.Println("Requirements:")
	fmt.Println("  pip install transformers safetensors torch")
	fmt.Println()
	fmt.Println("The download_model.py script will:")
	fmt.Println("  - Download the model from HuggingFace")
	fmt.Println("  - Convert to optimized safetensors format")
	fmt.Println("  - Create both config.json and model_info.json")
	fmt.Println("  - Handle tied weights properly")
}

func formatPrompt(question, chatFormat string) string {
	switch chatFormat {
	case "llama3":
		return fmt.Sprintf("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", question)
	case "falcon":
		return fmt.Sprintf("User: %s\nAssistant:", question)
	case "granite":
		systemMsg := "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
		return fmt.Sprintf("<|start_of_role|>system<|end_of_role|>%s<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>%s<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", systemMsg, question)
	default: // completion mode (GPT-2)
		return question
	}
}

func generateResponse(model *tensor.TransformerModel, promptTokens []int, tokenizer *purego.UniversalTokenizer, maxTokens int, temperature float64, useTemperature bool) ([]int, time.Duration, time.Duration) {
	// Initialize KV cache
	kvCache := tensor.NewKVCache(model.Config.NumLayers)

	// Prefill phase
	start := time.Now()
	logits, kvCache := model.ForwardWithCache(promptTokens, kvCache, 0)
	prefillTime := time.Since(start)

	// Get first token
	lastLogits := model.GetLogitsForLastToken(logits)

	var nextToken int
	if useTemperature {
		nextToken = sampleWithTemperature(lastLogits, temperature)
	} else {
		nextToken = argmax(lastLogits)
	}

	allTokens := append(promptTokens, nextToken)
	decoded, _ := tokenizer.Decode([]int{nextToken})
	fmt.Printf("Answer: %s", decoded)

	// Decode phase - generate remaining tokens
	decodeStart := time.Now()
	generatedTokens := []int{nextToken}
	for i := 0; i < maxTokens-1; i++ {
		logits, kvCache = model.ForwardWithCache([]int{allTokens[len(allTokens)-1]}, kvCache, len(allTokens)-1)
		lastLogits = model.GetLogitsForLastToken(logits)

		if useTemperature {
			nextToken = sampleWithTemperature(lastLogits, temperature)
		} else {
			nextToken = argmax(lastLogits)
		}

		decoded, _ = tokenizer.Decode([]int{nextToken})
		fmt.Printf("%s", decoded)

		allTokens = append(allTokens, nextToken)
		generatedTokens = append(generatedTokens, nextToken)

		if nextToken == tokenizer.EOSTokenID() {
			break
		}
	}
	decodeTime := time.Since(decodeStart)

	return generatedTokens, prefillTime, decodeTime
}

func encodeWithPython(modelDir, text string) ([]int, error) {
	cmd := exec.Command("python3", "scripts/encode_text.py", modelDir, text)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("python tokenizer failed: %v", err)
	}

	// Parse comma-separated token IDs
	tokenStr := strings.TrimSpace(string(output))
	tokenParts := strings.Split(tokenStr, ",")

	tokens := make([]int, 0, len(tokenParts))
	for _, part := range tokenParts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		tokenID, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("failed to parse token ID '%s': %v", part, err)
		}
		tokens = append(tokens, tokenID)
	}

	return tokens, nil
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

func sampleWithTemperature(logits []float32, temperature float64) int {
	// Find max logit for numerical stability
	var maxLogit float32 = logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	// Compute exp((logit - maxLogit) / temperature) and sum
	var sumExp float64
	probs := make([]float64, len(logits))
	for i, logit := range logits {
		expVal := math.Exp(float64(logit-maxLogit) / temperature)
		probs[i] = expVal
		sumExp += expVal
	}

	// Normalize to probabilities
	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample from categorical distribution
	r := rand.Float64()
	var cumSum float64
	for i, p := range probs {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}

	// Fallback to last token (shouldn't happen)
	return len(probs) - 1
}
