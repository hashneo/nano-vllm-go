package main

import (
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"

	"nano-vllm-go/purego/tensor"
)

func main() {
	modelDir := "./models/granite-350m"

	fmt.Println("Loading Granite model...")
	model, err := tensor.LoadModelFromDirectory(modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}

	// Same test as PyTorch
	prompt := "The capital of Germany is"
	fmt.Printf("\nPrompt: %s\n", prompt)

	// Encode
	tokens, err := encodeWithPython(modelDir, prompt)
	if err != nil {
		log.Fatalf("Failed to encode: %v", err)
	}
	fmt.Printf("Tokens: %v\n", tokens)

	// Forward pass
	kvCache := tensor.NewKVCache(model.Config.NumLayers)
	logits, _ := model.ForwardWithCache(tokens, kvCache, 0)

	// Get last token logits
	lastLogits := model.GetLogitsForLastToken(logits)

	// Find top 5
	type tokenScore struct {
		id    int
		score float32
	}
	var scores []tokenScore
	for i, score := range lastLogits {
		scores = append(scores, tokenScore{i, score})
	}

	// Sort by score (descending)
	for i := 0; i < len(scores); i++ {
		for j := i + 1; j < len(scores); j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	fmt.Printf("\nTop 5 predicted tokens:\n")
	for i := 0; i < 5; i++ {
		fmt.Printf("  %d. ID=%6d, score=%8.2f\n", i+1, scores[i].id, scores[i].score)
	}

	// Check for " Berlin" (ID 20437 according to PyTorch)
	berlinID := 20437
	berlinScore := lastLogits[berlinID]
	fmt.Printf("\nScore for ' Berlin' (ID=%d): %.2f\n", berlinID, berlinScore)
	fmt.Printf("PyTorch expected score: 35.47\n")

	// Check all logits stats
	var sum, minVal, maxVal float32
	minVal = lastLogits[0]
	maxVal = lastLogits[0]
	for _, v := range lastLogits {
		sum += v
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	mean := sum / float32(len(lastLogits))
	fmt.Printf("\nLogits stats: mean=%.2f, min=%.2f, max=%.2f\n", mean, minVal, maxVal)
}

func encodeWithPython(modelDir, text string) ([]int, error) {
	cmd := exec.Command("python3", "scripts/encode_text.py", modelDir, text)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("python tokenizer failed: %v", err)
	}

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
