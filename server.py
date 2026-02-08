#!/usr/bin/env python3
"""
Simple Python server for nano-vllm-go HTTP backend
This allows testing with real models without ONNX/LibTorch setup
"""

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

app = Flask(__name__)

# Global model and tokenizer
model = None
tokenizer = None

def load_model(model_name="Qwen/Qwen2-0.5B-Instruct"):
    """Load model and tokenizer"""
    global model, tokenizer

    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"  # Use "auto" for GPU
    )
    model.eval()

    print(f"✓ Model loaded successfully!")
    print(f"  Model type: {model.config.model_type}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token_id}")
    print()

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'model_type': model.config.model_type if hasattr(model.config, 'model_type') else 'unknown'
    })

@app.route('/tokenize', methods=['POST'])
def tokenize():
    """Tokenize text"""
    text = request.json.get('text', '')
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return jsonify({'tokens': tokens})

@app.route('/detokenize', methods=['POST'])
def detokenize():
    """Detokenize token IDs"""
    tokens = request.json.get('tokens', [])
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return jsonify({'text': text})

@app.route('/inference', methods=['POST'])
def inference():
    """Run inference on sequences"""
    data = request.json
    sequences = data.get('sequences', [])
    is_prefill = data.get('is_prefill', False)

    results = []

    for seq in sequences:
        token_ids = seq['token_ids']
        temperature = seq.get('temperature', 1.0)

        # Convert to tensor
        input_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Run model
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

        results.append(next_token)

    return jsonify({'token_ids': results})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Nano-vLLM-Go Inference Server')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2-0.5B-Instruct',
                        help='HuggingFace model name')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run server on')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to')

    args = parser.parse_args()

    # Load model
    try:
        load_model(args.model)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Start server
    print("="*70)
    print(f"Starting server on {args.host}:{args.port}")
    print("="*70)
    print()
    print("Ready to accept requests!")
    print()
    print("Test with:")
    print("  curl http://localhost:8000/health")
    print()

    app.run(host=args.host, port=args.port, threaded=False)

if __name__ == '__main__':
    main()
