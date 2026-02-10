package tensor

import (
	"math"
	"math/rand"
	"sort"
)

// SamplingParams holds parameters for token sampling
type SamplingParams struct {
	Temperature      float32
	TopP             float32 // Nucleus sampling
	TopK             int     // Top-k sampling
	RepetitionPenalty float32 // Penalty for repeating tokens (1.0 = no penalty, >1.0 = penalize)
}

// DefaultSamplingParams returns default sampling parameters
func DefaultSamplingParams() *SamplingParams {
	return &SamplingParams{
		Temperature:       1.0,
		TopP:              1.0,
		TopK:              0,   // 0 means disabled
		RepetitionPenalty: 1.2, // Slight penalty to reduce repetition
	}
}

// Sample samples a token from logits using the specified parameters
func Sample(logits []float32, params *SamplingParams) int {
	return SampleWithHistory(logits, nil, params)
}

// SampleWithHistory samples a token from logits with repetition penalty based on token history
func SampleWithHistory(logits []float32, previousTokens []int, params *SamplingParams) int {
	if params == nil {
		params = DefaultSamplingParams()
	}

	// Make a copy of logits to avoid modifying the original
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	// Apply repetition penalty
	if params.RepetitionPenalty != 1.0 && len(previousTokens) > 0 {
		// Track which tokens have appeared and how recently
		tokenCounts := make(map[int]int)
		for i, token := range previousTokens {
			// Give more weight to recent tokens
			weight := 1
			if i >= len(previousTokens)-10 { // Last 10 tokens get higher weight
				weight = 3
			}
			tokenCounts[token] += weight
		}

		// Apply penalty to repeated tokens
		for token, count := range tokenCounts {
			if token < len(logitsCopy) {
				// Penalize based on frequency
				penalty := params.RepetitionPenalty * float32(count)
				if logitsCopy[token] > 0 {
					logitsCopy[token] /= penalty
				} else {
					logitsCopy[token] *= penalty
				}
			}
		}
	}

	// Apply temperature
	if params.Temperature > 0 && params.Temperature != 1.0 {
		for i := range logitsCopy {
			logitsCopy[i] /= params.Temperature
		}
	}

	// Convert logits to probabilities using softmax
	probs := softmax(logitsCopy)

	// Apply top-k filtering
	if params.TopK > 0 && params.TopK < len(probs) {
		probs = topKFiltering(probs, params.TopK)
	}

	// Apply top-p (nucleus) filtering
	if params.TopP < 1.0 {
		probs = topPFiltering(probs, params.TopP)
	}

	// Renormalize probabilities
	sum := float32(0)
	for _, p := range probs {
		sum += p
	}
	if sum > 0 {
		for i := range probs {
			probs[i] /= sum
		}
	}

	// Sample from the distribution
	return sampleMultinomial(probs)
}

// softmax converts logits to probabilities
func softmax(logits []float32) []float32 {
	// Find max for numerical stability
	maxLogit := logits[0]
	for _, l := range logits[1:] {
		if l > maxLogit {
			maxLogit = l
		}
	}

	// Compute exp and sum
	probs := make([]float32, len(logits))
	sum := float32(0)
	for i, l := range logits {
		probs[i] = float32(math.Exp(float64(l - maxLogit)))
		sum += probs[i]
	}

	// Normalize
	for i := range probs {
		probs[i] /= sum
	}

	return probs
}

// topKFiltering keeps only top-k probabilities, zeros out the rest
func topKFiltering(probs []float32, k int) []float32 {
	type indexedProb struct {
		idx  int
		prob float32
	}

	// Create indexed array
	indexed := make([]indexedProb, len(probs))
	for i, p := range probs {
		indexed[i] = indexedProb{i, p}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Zero out everything beyond top-k
	result := make([]float32, len(probs))
	for i := 0; i < k && i < len(indexed); i++ {
		result[indexed[i].idx] = indexed[i].prob
	}

	return result
}

// topPFiltering keeps only probabilities that sum to >= p (nucleus sampling)
func topPFiltering(probs []float32, p float32) []float32 {
	type indexedProb struct {
		idx  int
		prob float32
	}

	// Create indexed array
	indexed := make([]indexedProb, len(probs))
	for i, prob := range probs {
		indexed[i] = indexedProb{i, prob}
	}

	// Sort by probability descending
	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].prob > indexed[j].prob
	})

	// Find cutoff where cumulative probability >= p
	cumProb := float32(0)
	cutoff := len(indexed)
	for i, item := range indexed {
		cumProb += item.prob
		if cumProb >= p {
			cutoff = i + 1
			break
		}
	}

	// Zero out everything beyond cutoff
	result := make([]float32, len(probs))
	for i := 0; i < cutoff; i++ {
		result[indexed[i].idx] = indexed[i].prob
	}

	return result
}

// sampleMultinomial samples from a probability distribution
func sampleMultinomial(probs []float32) int {
	// Compute cumulative probabilities
	cumProbs := make([]float32, len(probs))
	cumProbs[0] = probs[0]
	for i := 1; i < len(probs); i++ {
		cumProbs[i] = cumProbs[i-1] + probs[i]
	}

	// Sample uniform random number
	r := rand.Float32() * cumProbs[len(cumProbs)-1]

	// Binary search to find the sampled index
	idx := sort.Search(len(cumProbs), func(i int) bool {
		return cumProbs[i] >= r
	})

	if idx >= len(probs) {
		idx = len(probs) - 1
	}

	return idx
}
