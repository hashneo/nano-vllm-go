#!/bin/bash
# Quick start script for Pure Go Transformer (Native Mode)

set -e

echo "========================================="
echo "Nano-vLLM-Go - Pure Go Transformer Mode"
echo "========================================="
echo ""

# Check if model exists
MODEL_DIR="./models/gpt2"
MODEL_PATH="$MODEL_DIR/model.safetensors"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Downloading GPT-2..."
    echo ""

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found"
        echo "Please install Python 3.8+ to download the model"
        exit 1
    fi

    # Install dependencies
    echo "Installing Python dependencies..."
    pip3 install -q torch transformers safetensors 2>/dev/null || {
        echo "⚠️  Warning: Could not install Python packages quietly"
        pip3 install torch transformers safetensors
    }

    # Download model
    echo ""
    echo "Downloading GPT-2 (this will take a few minutes)..."
    python3 scripts/download_gpt2.py --model gpt2 --output "$MODEL_DIR" || {
        echo "❌ Failed to download model"
        exit 1
    }

    echo ""
fi

echo "✓ Model found: $MODEL_PATH"
echo ""

# Build Go program
echo "Building Go program..."
go build -o bin/native_test ./purego/example_native || {
    echo "❌ Build failed"
    exit 1
}
echo "✓ Build complete"
echo ""

# Get prompt from args or use default
if [ $# -eq 0 ]; then
    PROMPT="Once upon a time"
else
    PROMPT="$*"
fi

# Run
echo "========================================="
echo "Running Pure Go Transformer"
echo "========================================="
echo ""

export MODEL_PATH="$MODEL_PATH"
export TOKENIZER_PATH="$MODEL_DIR"

./bin/native_test "$PROMPT"

echo ""
echo "========================================="
echo "💡 Tips:"
echo "  • Pure Go = No dependencies at runtime"
echo "  • Speed: ~5-10 tok/s (educational)"
echo "  • Perfect for learning & experiments"
echo ""
echo "Try another prompt:"
echo "  $0 \"Your custom prompt here\""
echo "========================================="
