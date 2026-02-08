package purego

import (
	"fmt"
	"math"
	"math/rand"

	ort "github.com/yalue/onnxruntime_go"
	"nano-vllm-go/nanovllm"
)

// ONNXModelRunner implements ModelRunner using ONNX Runtime
type ONNXModelRunner struct {
	modelPath   string
	config      *nanovllm.Config
	vocabSize   int
	hiddenSize  int
	initialized bool
}

// NewONNXModelRunner creates a new ONNX-based model runner
func NewONNXModelRunner(modelPath string, config *nanovllm.Config) (*ONNXModelRunner, error) {
	// Initialize ONNX Runtime
	if !ort.IsInitialized() {
		err := ort.InitializeEnvironment()
		if err != nil {
			return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
		}
	}

	runner := &ONNXModelRunner{
		modelPath:   modelPath,
		config:      config,
		vocabSize:   32000, // Will be set by tokenizer
		hiddenSize:  4096,
		initialized: true,
	}

	fmt.Printf("✓ ONNX runtime initialized (model will be loaded per-request)\n")
	return runner, nil
}

// Run executes inference on the sequences
func (m *ONNXModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	if !m.initialized {
		return nil, fmt.Errorf("model runner not initialized")
	}

	batchSize := len(seqs)
	if batchSize == 0 {
		return nil, fmt.Errorf("no sequences to process")
	}

	tokenIDs := make([]int, batchSize)

	// Create session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer options.Destroy()

	// Enable CPU optimizations
	if err := options.SetIntraOpNumThreads(4); err != nil {
		return nil, fmt.Errorf("failed to set threads: %w", err)
	}

	// Process each sequence (for now, one at a time)
	for i, seq := range seqs {
		// Prepare input tensor
		inputIDs := seq.TokenIDs
		if len(inputIDs) == 0 {
			return nil, fmt.Errorf("sequence %d has no tokens", seq.SeqID)
		}

		// Create input shape and data
		inputShape := ort.NewShape(1, int64(len(inputIDs)))
		inputData := make([]int64, len(inputIDs))
		for j, id := range inputIDs {
			inputData[j] = int64(id)
		}

		// Create input tensor
		inputTensor, err := ort.NewTensor(inputShape, inputData)
		if err != nil {
			return nil, fmt.Errorf("failed to create input tensor: %w", err)
		}
		defer inputTensor.Destroy()

		// Create output tensor (shape will be [1, seq_len, vocab_size])
		outputShape := ort.NewShape(1, int64(len(inputIDs)), int64(m.vocabSize))
		outputData := make([]float32, len(inputIDs)*m.vocabSize)
		outputTensor, err := ort.NewTensor(outputShape, outputData)
		if err != nil {
			inputTensor.Destroy()
			return nil, fmt.Errorf("failed to create output tensor: %w", err)
		}
		defer outputTensor.Destroy()

		// Create session with pre-allocated tensors
		session, err := ort.NewAdvancedSession(
			m.modelPath,
			[]string{"input_ids"},
			[]string{"logits"},
			[]ort.Value{inputTensor},
			[]ort.Value{outputTensor},
			options,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create session: %w", err)
		}
		defer session.Destroy()

		// Run inference (updates outputTensor in-place)
		if err := session.Run(); err != nil {
			return nil, fmt.Errorf("inference failed: %w", err)
		}

		// Get logits for last token
		logits := outputTensor.GetData()
		seqLen := len(inputIDs)
		lastTokenStart := (seqLen - 1) * m.vocabSize
		lastTokenLogits := logits[lastTokenStart : lastTokenStart+m.vocabSize]

		// Sample token
		tokenID := m.sampleToken(lastTokenLogits, seq.Temperature)
		tokenIDs[i] = tokenID
	}

	return tokenIDs, nil
}

// sampleToken samples a token from logits using temperature sampling
func (m *ONNXModelRunner) sampleToken(logits []float32, temperature float64) int {
	// Make a copy to avoid modifying original
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	// Apply temperature
	if temperature != 1.0 {
		for i := range logitsCopy {
			logitsCopy[i] /= float32(temperature)
		}
	}

	// Compute softmax
	maxLogit := logitsCopy[0]
	for _, logit := range logitsCopy {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float32
	probs := make([]float32, len(logitsCopy))
	for i, logit := range logitsCopy {
		probs[i] = float32(math.Exp(float64(logit - maxLogit)))
		sumExp += probs[i]
	}

	for i := range probs {
		probs[i] /= sumExp
	}

	// Sample from distribution
	r := rand.Float32()
	var cumProb float32
	for i, prob := range probs {
		cumProb += prob
		if r <= cumProb {
			return i
		}
	}

	return len(probs) - 1
}

// Close cleans up resources
func (m *ONNXModelRunner) Close() error {
	m.initialized = false
	// Note: We destroy the environment in a separate cleanup if needed
	return nil
}

// GetVocabSize returns the vocabulary size
func (m *ONNXModelRunner) GetVocabSize() int {
	return m.vocabSize
}

// SetVocabSize sets the vocabulary size
func (m *ONNXModelRunner) SetVocabSize(size int) {
	m.vocabSize = size
}
