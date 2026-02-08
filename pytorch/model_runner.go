//go:build pytorch
// +build pytorch

package pytorch

/*
#cgo CFLAGS: -I${SRCDIR}/../third_party/libtorch/include -I${SRCDIR}/../third_party/libtorch/include/torch/csrc/api/include
#cgo LDFLAGS: -L${SRCDIR}/../third_party/libtorch/lib -ltorch -ltorch_cpu -lc10
#include <torch/torch.h>
#include <torch/script.h>

// C wrapper functions for PyTorch
extern void* load_model(const char* path);
extern void free_model(void* model);
extern void* run_inference(void* model, long* input_ids, int batch_size, int seq_len, float temperature);
extern long sample_token(void* logits_ptr, int vocab_size, float temperature);
*/
import "C"
import (
	"fmt"
	"unsafe"

	"nano-vllm-go/nanovllm"
)

// PyTorchModelRunner implements ModelRunner using LibTorch (PyTorch C++ API)
type PyTorchModelRunner struct {
	modelPtr  unsafe.Pointer
	config    *nanovllm.Config
	vocabSize int
}

// NewPyTorchModelRunner creates a new PyTorch-based model runner
func NewPyTorchModelRunner(modelPath string, config *nanovllm.Config) (*PyTorchModelRunner, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	modelPtr := C.load_model(cPath)
	if modelPtr == nil {
		return nil, fmt.Errorf("failed to load PyTorch model from %s", modelPath)
	}

	return &PyTorchModelRunner{
		modelPtr:  modelPtr,
		config:    config,
		vocabSize: 32000, // Default, should be loaded from config
	}, nil
}

// Run executes inference using PyTorch
func (m *PyTorchModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
	batchSize := len(seqs)
	if batchSize == 0 {
		return nil, fmt.Errorf("no sequences to process")
	}

	// Determine max sequence length
	var maxLen int
	if isPrefill {
		for _, seq := range seqs {
			if seq.Len() > maxLen {
				maxLen = seq.Len()
			}
		}
	} else {
		maxLen = 1
	}

	// Prepare input tensor
	inputData := make([]C.long, batchSize*maxLen)
	for i, seq := range seqs {
		var tokens []int
		if isPrefill {
			tokens = seq.TokenIDs[:seq.Len()]
		} else {
			tokens = []int{seq.LastToken}
		}

		for j, token := range tokens {
			inputData[i*maxLen+j] = C.long(token)
		}
	}

	// Run inference through C wrapper
	logitsPtr := C.run_inference(
		m.modelPtr,
		&inputData[0],
		C.int(batchSize),
		C.int(maxLen),
		C.float(seqs[0].Temperature),
	)
	if logitsPtr == nil {
		return nil, fmt.Errorf("inference failed")
	}

	// Sample tokens
	tokenIDs := make([]int, batchSize)
	for i := 0; i < batchSize; i++ {
		temperature := seqs[i].Temperature
		tokenID := C.sample_token(logitsPtr, C.int(m.vocabSize), C.float(temperature))
		tokenIDs[i] = int(tokenID)
	}

	return tokenIDs, nil
}

// Close cleans up PyTorch resources
func (m *PyTorchModelRunner) Close() error {
	if m.modelPtr != nil {
		C.free_model(m.modelPtr)
		m.modelPtr = nil
	}
	return nil
}

// SetVocabSize sets the vocabulary size
func (m *PyTorchModelRunner) SetVocabSize(size int) {
	m.vocabSize = size
}
