package tensor

import (
	"math"
	"sort"
)

// MoELayer represents a Mixture of Experts layer
type MoELayer struct {
	// Router selects which experts to use
	Router *Tensor // [hidden, num_experts]

	// Experts (shared-weight MoE: all experts share same weights)
	InputLinear  *Tensor // [hidden, intermediate] or [num_experts, hidden, intermediate] for separate experts
	OutputLinear *Tensor // [intermediate, hidden] or [num_experts, intermediate, hidden] for separate experts

	// Configuration
	NumExperts       int
	NumExpertsPerTok int // Top-k experts to use
	IntermediateSize int
	HiddenSize       int
	SharedExperts    bool // If true, all experts share the same weights
}

// NewMoELayer creates a new MoE layer
func NewMoELayer(numExperts, numExpertsPerTok, hiddenSize, intermediateSize int) *MoELayer {
	return &MoELayer{
		NumExperts:       numExperts,
		NumExpertsPerTok: numExpertsPerTok,
		HiddenSize:       hiddenSize,
		IntermediateSize: intermediateSize,
		SharedExperts:    true, // Default to shared weights
	}
}

// expertScore represents an expert with its routing score
type expertScore struct {
	index int
	score float32
}

// Forward performs the forward pass through the MoE layer
func (m *MoELayer) Forward(x *Tensor) *Tensor {
	// Handle both 2D [batch*seq_len, hidden] and 3D [batch, seq_len, hidden] inputs
	var batchSeqLen, hiddenSize int
	var needsReshape bool
	var originalShape []int

	if len(x.Shape) == 3 {
		// Input is 3D [batch, seq_len, hidden], need to flatten
		needsReshape = true
		originalShape = x.Shape
		batchSeqLen = x.Shape[0] * x.Shape[1]
		hiddenSize = x.Shape[2]
		x = x.Reshape(batchSeqLen, hiddenSize)
	} else {
		// Input is already 2D [batch*seq_len, hidden]
		batchSeqLen = x.Shape[0]
		hiddenSize = x.Shape[1]
	}

	// 1. Compute router logits: [batch*seq, num_experts]
	routerLogits := MatMul(x, m.Router)

	// 2. Apply softmax to get routing probabilities
	routerProbs := Softmax(routerLogits)

	// 3. Initialize output tensor
	output := NewTensor(batchSeqLen, hiddenSize)

	// 4. For each token, select top-k experts and compute their outputs
	for i := 0; i < batchSeqLen; i++ {
		// Get routing probabilities for this token
		tokenProbs := make([]expertScore, m.NumExperts)
		for j := 0; j < m.NumExperts; j++ {
			tokenProbs[j] = expertScore{
				index: j,
				score: routerProbs.Data[i*m.NumExperts+j],
			}
		}

		// Sort by score (descending) and take top-k
		sort.Slice(tokenProbs, func(a, b int) bool {
			return tokenProbs[a].score > tokenProbs[b].score
		})

		// Normalize top-k scores to sum to 1
		topK := tokenProbs[:m.NumExpertsPerTok]
		sumScores := float32(0)
		for _, exp := range topK {
			sumScores += exp.score
		}

		// Extract token embedding: [hidden]
		tokenInput := make([]float32, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			tokenInput[j] = x.Data[i*hiddenSize+j]
		}

		// Process through selected experts
		for _, exp := range topK {
			expertIdx := exp.index
			weight := exp.score / sumScores // Normalized weight

			// Forward through expert
			var expertOutput []float32
			if m.SharedExperts {
				// All experts share the same weights
				expertOutput = m.forwardExpertShared(tokenInput)
			} else {
				// Each expert has separate weights
				expertOutput = m.forwardExpertSeparate(tokenInput, expertIdx)
			}

			// Add weighted expert output to final output
			for j := 0; j < hiddenSize; j++ {
				output.Data[i*hiddenSize+j] += weight * expertOutput[j]
			}
		}
	}

	// Reshape back to 3D if needed
	if needsReshape {
		output = output.Reshape(originalShape...)
	}

	return output
}

// forwardExpertShared processes input through a shared-weight expert
func (m *MoELayer) forwardExpertShared(input []float32) []float32 {
	// input: [hidden]
	// InputLinear: [hidden, intermediate]
	// OutputLinear: [intermediate, hidden]

	// 1. Project to intermediate: hidden @ [hidden, intermediate] -> [intermediate]
	intermediate := make([]float32, m.IntermediateSize)
	for i := 0; i < m.IntermediateSize; i++ {
		sum := float32(0)
		for j := 0; j < m.HiddenSize; j++ {
			sum += input[j] * m.InputLinear.Data[j*m.IntermediateSize+i]
		}
		intermediate[i] = sum
	}

	// 2. Apply SiLU activation (x * sigmoid(x))
	for i := 0; i < m.IntermediateSize; i++ {
		x := intermediate[i]
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-x))))
		intermediate[i] = x * sigmoid
	}

	// 3. Project back to hidden: intermediate @ [intermediate, hidden] -> [hidden]
	output := make([]float32, m.HiddenSize)
	for i := 0; i < m.HiddenSize; i++ {
		sum := float32(0)
		for j := 0; j < m.IntermediateSize; j++ {
			sum += intermediate[j] * m.OutputLinear.Data[j*m.HiddenSize+i]
		}
		output[i] = sum
	}

	return output
}

// forwardExpertSeparate processes input through an expert with separate weights
func (m *MoELayer) forwardExpertSeparate(input []float32, expertIdx int) []float32 {
	// Granite MoE uses GLU-style gating:
	// 1. Project input with InputLinear: [hidden] -> [2 * intermediate]
	// 2. Split into gate and up: [2 * intermediate] -> [intermediate], [intermediate]
	// 3. Apply activation(gate) * up
	// 4. Project with OutputLinear: [intermediate] -> [hidden]
	//
	// Weights shapes:
	// InputLinear: [num_experts, 1024, 1024] where 1024 output = 2 * 512
	// OutputLinear: [num_experts, 1024, 512]

	inputShape := m.InputLinear.Shape   // [32, 1024, 1024]
	outputShape := m.OutputLinear.Shape // [32, 1024, 512]

	inputOut := inputShape[1]      // 1024 (2 * intermediate_size)
	inputIn := inputShape[2]       // 1024 (hidden)
	outputOut := outputShape[1]    // 1024 (hidden)
	outputIn := outputShape[2]     // 512 (intermediate_size)

	// Calculate expert offset
	expertInOffset := expertIdx * inputOut * inputIn
	expertOutOffset := expertIdx * outputOut * outputIn

	// 1. First projection: [hidden=1024] @ [1024, 1024]^T = [1024]
	proj1 := make([]float32, inputOut)
	for i := 0; i < inputOut; i++ {
		sum := float32(0)
		for j := 0; j < inputIn; j++ {
			weightIdx := expertInOffset + i*inputIn + j
			sum += input[j] * m.InputLinear.Data[weightIdx]
		}
		proj1[i] = sum
	}

	// 2. Split into gate (first 512) and up (last 512), apply GLU
	// GLU: activation(gate) * up
	intermediate := make([]float32, outputIn)
	for i := 0; i < outputIn; i++ {
		gate := proj1[i]
		up := proj1[i+outputIn]
		// Apply SiLU to gate
		sigmoid := 1.0 / (1.0 + float32(math.Exp(float64(-gate))))
		gateActivated := gate * sigmoid
		// Multiply by up
		intermediate[i] = gateActivated * up
	}

	// 3. Output projection: [512] @ [1024, 512]^T -> [1024]
	output := make([]float32, outputOut)
	for i := 0; i < outputOut; i++ {
		sum := float32(0)
		for j := 0; j < outputIn; j++ {
			weightIdx := expertOutOffset + i*outputIn + j
			sum += intermediate[j] * m.OutputLinear.Data[weightIdx]
		}
		output[i] = sum
	}

	return output
}
