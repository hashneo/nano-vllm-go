package nanovllm

import (
	"fmt"
	"os"
)

// Config holds the configuration for the LLM engine
type Config struct {
	Model                string
	MaxNumBatchedTokens  int
	MaxNumSeqs           int
	MaxModelLen          int
	GPUMemoryUtilization float64
	TensorParallelSize   int
	EnforceEager         bool
	EOS                  int
	KVCacheBlockSize     int
	NumKVCacheBlocks     int
}

// ConfigOption is a functional option for Config
type ConfigOption func(*Config)

// NewConfig creates a new Config with default values
func NewConfig(modelPath string, opts ...ConfigOption) *Config {
	c := &Config{
		Model:                modelPath,
		MaxNumBatchedTokens:  16384,
		MaxNumSeqs:           512,
		MaxModelLen:          4096,
		GPUMemoryUtilization: 0.9,
		TensorParallelSize:   1,
		EnforceEager:         false,
		EOS:                  -1,
		KVCacheBlockSize:     256,
		NumKVCacheBlocks:     -1,
	}

	for _, opt := range opts {
		opt(c)
	}

	if err := c.validate(); err != nil {
		panic(err)
	}

	return c
}

// validate checks if the configuration is valid
func (c *Config) validate() error {
	if _, err := os.Stat(c.Model); os.IsNotExist(err) {
		return fmt.Errorf("model directory does not exist: %s", c.Model)
	}

	if c.KVCacheBlockSize%256 != 0 {
		return fmt.Errorf("kvcache_block_size must be divisible by 256")
	}

	if c.TensorParallelSize < 1 || c.TensorParallelSize > 8 {
		return fmt.Errorf("tensor_parallel_size must be between 1 and 8")
	}

	if c.MaxNumBatchedTokens < c.MaxModelLen {
		return fmt.Errorf("max_num_batched_tokens must be >= max_model_len")
	}

	return nil
}

// WithMaxNumBatchedTokens sets the maximum number of batched tokens
func WithMaxNumBatchedTokens(n int) ConfigOption {
	return func(c *Config) {
		c.MaxNumBatchedTokens = n
	}
}

// WithMaxNumSeqs sets the maximum number of sequences
func WithMaxNumSeqs(n int) ConfigOption {
	return func(c *Config) {
		c.MaxNumSeqs = n
	}
}

// WithMaxModelLen sets the maximum model length
func WithMaxModelLen(n int) ConfigOption {
	return func(c *Config) {
		c.MaxModelLen = n
	}
}

// WithGPUMemoryUtilization sets the GPU memory utilization
func WithGPUMemoryUtilization(f float64) ConfigOption {
	return func(c *Config) {
		c.GPUMemoryUtilization = f
	}
}

// WithTensorParallelSize sets the tensor parallel size
func WithTensorParallelSize(n int) ConfigOption {
	return func(c *Config) {
		c.TensorParallelSize = n
	}
}

// WithEnforceEager sets whether to enforce eager mode
func WithEnforceEager(b bool) ConfigOption {
	return func(c *Config) {
		c.EnforceEager = b
	}
}

// WithEOS sets the EOS token ID
func WithEOS(id int) ConfigOption {
	return func(c *Config) {
		c.EOS = id
	}
}

// WithKVCacheBlockSize sets the KV cache block size
func WithKVCacheBlockSize(n int) ConfigOption {
	return func(c *Config) {
		c.KVCacheBlockSize = n
	}
}

// WithNumKVCacheBlocks sets the number of KV cache blocks
func WithNumKVCacheBlocks(n int) ConfigOption {
	return func(c *Config) {
		c.NumKVCacheBlocks = n
	}
}
