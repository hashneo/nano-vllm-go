# Integration Guide

This guide shows how to integrate nano-vllm-go with real model inference backends.

## Option 1: HTTP/gRPC Inference Server

The simplest approach - call a Python-based inference server.

### Python Server (using original nano-vllm)

```python
# server.py
from flask import Flask, request, jsonify
from nanovllm import LLM, SamplingParams

app = Flask(__name__)
llm = LLM("/path/to/model", enforce_eager=True)

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    token_ids = data['token_ids']
    is_prefill = data['is_prefill']
    temperature = data['temperature']

    # Run model
    # ... actual inference logic ...

    return jsonify({'token_ids': result_tokens})

if __name__ == '__main__':
    app.run(port=8000)
```

### Go Client

```go
package main

import (
    "bytes"
    "encoding/json"
    "net/http"
    "github.com/your-username/nano-vllm-go/nanovllm"
)

type HTTPModelRunner struct {
    serverURL string
    client    *http.Client
}

func NewHTTPModelRunner(serverURL string) *HTTPModelRunner {
    return &HTTPModelRunner{
        serverURL: serverURL,
        client:    &http.Client{},
    }
}

func (m *HTTPModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    // Prepare request
    type Request struct {
        TokenIDs  [][]int `json:"token_ids"`
        IsPrefill bool    `json:"is_prefill"`
    }

    tokenIDs := make([][]int, len(seqs))
    for i, seq := range seqs {
        tokenIDs[i] = seq.TokenIDs
    }

    reqBody := Request{
        TokenIDs:  tokenIDs,
        IsPrefill: isPrefill,
    }

    // Send HTTP request
    body, _ := json.Marshal(reqBody)
    resp, err := m.client.Post(m.serverURL+"/inference", "application/json", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    // Parse response
    var result struct {
        TokenIDs []int `json:"token_ids"`
    }
    json.NewDecoder(resp.Body).Decode(&result)

    return result.TokenIDs, nil
}

func (m *HTTPModelRunner) Close() error {
    return nil
}

// Usage
func main() {
    config := nanovllm.NewConfig(".")
    modelRunner := NewHTTPModelRunner("http://localhost:8000")
    tokenizer := NewHTTPTokenizer("http://localhost:8000") // Similar implementation

    llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
    defer llm.Close()

    // Use llm...
}
```

## Option 2: CGo with ONNX Runtime

Use ONNX Runtime C API through CGo.

### Installation

```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar xzf onnxruntime-linux-x64-1.16.0.tgz
export LD_LIBRARY_PATH=$(pwd)/onnxruntime-linux-x64-1.16.0/lib:$LD_LIBRARY_PATH
```

### Go Implementation

```go
package main

/*
#cgo CFLAGS: -I/path/to/onnxruntime/include
#cgo LDFLAGS: -L/path/to/onnxruntime/lib -lonnxruntime
#include <onnxruntime_c_api.h>
#include <stdlib.h>
*/
import "C"
import (
    "unsafe"
    "github.com/your-username/nano-vllm-go/nanovllm"
)

type ONNXModelRunner struct {
    session *C.OrtSession
    // ... other fields
}

func NewONNXModelRunner(modelPath string) (*ONNXModelRunner, error) {
    // Initialize ONNX Runtime
    // Load model
    // Return runner
    return &ONNXModelRunner{}, nil
}

func (m *ONNXModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    // Prepare input tensors
    // Run inference via ONNX Runtime C API
    // Extract output tokens
    // Return results
    return nil, nil
}

func (m *ONNXModelRunner) Close() error {
    // Clean up ONNX Runtime resources
    return nil
}
```

## Option 3: Pure Go with ONNX Go

Use a pure Go ONNX runtime (slower but no CGo complexity).

```go
package main

import (
    "github.com/owulveryck/onnx-go"
    "github.com/your-username/nano-vllm-go/nanovllm"
)

type PureGoModelRunner struct {
    model *onnx.Model
}

func NewPureGoModelRunner(modelPath string) (*PureGoModelRunner, error) {
    // Load ONNX model
    model := onnx.NewModel()
    // ... load model ...

    return &PureGoModelRunner{
        model: model,
    }, nil
}

func (m *PureGoModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    // Prepare tensors
    // Run model
    // Sample tokens
    return nil, nil
}
```

## Option 4: Llamafile Integration

Use Mozilla's llamafile for easy model deployment.

```bash
# Download llamafile
wget https://github.com/Mozilla-Ocho/llamafile/releases/download/0.8.4/llamafile-0.8.4

# Run model server
chmod +x llamafile-0.8.4
./llamafile-0.8.4 -m model.gguf --server --port 8080
```

```go
// Use HTTP client similar to Option 1
modelRunner := NewHTTPModelRunner("http://localhost:8080")
```

## Tokenizer Integration

### Option 1: HTTP API to Hugging Face

```go
type HTTPTokenizer struct {
    serverURL string
    eosID     int
}

func (t *HTTPTokenizer) Encode(text string) ([]int, error) {
    // POST to tokenizer server
    // Return token IDs
}

func (t *HTTPTokenizer) Decode(tokenIDs []int) (string, error) {
    // POST to tokenizer server
    // Return text
}
```

### Option 2: SentencePiece Go Bindings

```go
import "github.com/yoheimuta/go-sentencepiece"

type SPTokenizer struct {
    processor *sentencepiece.Processor
    eosID     int
}

func NewSPTokenizer(modelPath string) (*SPTokenizer, error) {
    processor, err := sentencepiece.NewProcessor(modelPath)
    if err != nil {
        return nil, err
    }
    return &SPTokenizer{
        processor: processor,
        eosID:     2,
    }, nil
}

func (t *SPTokenizer) Encode(text string) ([]int, error) {
    ids := t.processor.Encode(text)
    return ids, nil
}

func (t *SPTokenizer) Decode(tokenIDs []int) (string, error) {
    text := t.processor.Decode(tokenIDs)
    return text, nil
}
```

### Option 3: TikToken for OpenAI Models

```go
import "github.com/pkoukk/tiktoken-go"

type TikTokenizer struct {
    encoding *tiktoken.Tiktoken
    eosID    int
}

func NewTikTokenizer(modelName string) (*TikTokenizer, error) {
    encoding, err := tiktoken.EncodingForModel(modelName)
    if err != nil {
        return nil, err
    }
    return &TikTokenizer{
        encoding: encoding,
        eosID:    50256,
    }, nil
}

func (t *TikTokenizer) Encode(text string) ([]int, error) {
    return t.encoding.Encode(text, nil, nil), nil
}

func (t *TikTokenizer) Decode(tokenIDs []int) (string, error) {
    return t.encoding.Decode(tokenIDs), nil
}
```

## Complete Example: HTTP Backend

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "log"
    "net/http"

    "github.com/your-username/nano-vllm-go/nanovllm"
)

// HTTP-based model runner
type HTTPModelRunner struct {
    serverURL string
    client    *http.Client
}

func NewHTTPModelRunner(serverURL string) *HTTPModelRunner {
    return &HTTPModelRunner{
        serverURL: serverURL,
        client:    &http.Client{},
    }
}

func (m *HTTPModelRunner) Run(seqs []*nanovllm.Sequence, isPrefill bool) ([]int, error) {
    type Request struct {
        Sequences []struct {
            TokenIDs []int   `json:"token_ids"`
            Temp     float64 `json:"temperature"`
        } `json:"sequences"`
        IsPrefill bool `json:"is_prefill"`
    }

    req := Request{
        Sequences: make([]struct {
            TokenIDs []int   `json:"token_ids"`
            Temp     float64 `json:"temperature"`
        }, len(seqs)),
        IsPrefill: isPrefill,
    }

    for i, seq := range seqs {
        req.Sequences[i].TokenIDs = seq.TokenIDs
        req.Sequences[i].Temp = seq.Temperature
    }

    body, _ := json.Marshal(req)
    resp, err := m.client.Post(m.serverURL+"/v1/inference", "application/json", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        TokenIDs []int `json:"token_ids"`
    }
    json.NewDecoder(resp.Body).Decode(&result)

    return result.TokenIDs, nil
}

func (m *HTTPModelRunner) Close() error {
    return nil
}

// HTTP-based tokenizer
type HTTPTokenizer struct {
    serverURL string
    eosID     int
}

func NewHTTPTokenizer(serverURL string, eosID int) *HTTPTokenizer {
    return &HTTPTokenizer{
        serverURL: serverURL,
        eosID:     eosID,
    }
}

func (t *HTTPTokenizer) Encode(text string) ([]int, error) {
    type Request struct {
        Text string `json:"text"`
    }

    req := Request{Text: text}
    body, _ := json.Marshal(req)

    resp, err := http.Post(t.serverURL+"/v1/tokenize", "application/json", bytes.NewReader(body))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        TokenIDs []int `json:"token_ids"`
    }
    json.NewDecoder(resp.Body).Decode(&result)

    return result.TokenIDs, nil
}

func (t *HTTPTokenizer) Decode(tokenIDs []int) (string, error) {
    type Request struct {
        TokenIDs []int `json:"token_ids"`
    }

    req := Request{TokenIDs: tokenIDs}
    body, _ := json.Marshal(req)

    resp, err := http.Post(t.serverURL+"/v1/detokenize", "application/json", bytes.NewReader(body))
    if err != nil {
        return "", err
    }
    defer resp.Body.Close()

    var result struct {
        Text string `json:"text"`
    }
    json.NewDecoder(resp.Body).Decode(&result)

    return result.Text, nil
}

func (t *HTTPTokenizer) EOSTokenID() int {
    return t.eosID
}

// Main usage
func main() {
    serverURL := "http://localhost:8000"

    config := nanovllm.NewConfig(
        ".",
        nanovllm.WithEOS(2),
    )

    modelRunner := NewHTTPModelRunner(serverURL)
    tokenizer := NewHTTPTokenizer(serverURL, 2)

    llm := nanovllm.NewLLMWithComponents(config, modelRunner, tokenizer)
    defer llm.Close()

    samplingParams := nanovllm.NewSamplingParams(
        nanovllm.WithTemperature(0.7),
        nanovllm.WithMaxTokens(100),
    )

    prompts := []string{
        "Hello, world!",
        "Explain quantum computing.",
    }

    outputs, err := llm.GenerateSimple(prompts, samplingParams, true)
    if err != nil {
        log.Fatalf("Generation failed: %v", err)
    }

    for i, output := range outputs {
        fmt.Printf("Prompt: %s\n", prompts[i])
        fmt.Printf("Output: %s\n\n", output.Text)
    }
}
```

## Performance Tips

1. **Batch HTTP Requests**: Send multiple sequences in one HTTP call
2. **Connection Pooling**: Reuse HTTP connections
3. **Model Quantization**: Use INT8/INT4 models for faster inference
4. **GPU Batching**: Let the inference server handle GPU batching
5. **Tensor Parallelism**: Split large models across multiple GPUs

## Production Checklist

- [ ] Implement proper error handling
- [ ] Add request timeouts
- [ ] Implement retry logic
- [ ] Add monitoring and metrics
- [ ] Set up health checks
- [ ] Configure resource limits
- [ ] Enable logging
- [ ] Add authentication for HTTP endpoints
- [ ] Implement rate limiting
- [ ] Set up load balancing for multiple inference servers

## Next Steps

1. Choose your inference backend (HTTP recommended for simplicity)
2. Implement ModelRunner and Tokenizer interfaces
3. Test with small models first
4. Tune batch size and memory settings
5. Add production features (monitoring, logging, etc.)
6. Deploy and scale
