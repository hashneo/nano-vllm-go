#!/bin/bash
# Quick start script for ONNX Runtime Mode

set -e

echo "========================================="
echo "Nano-vLLM-Go - ONNX Runtime Mode"
echo "========================================="
echo ""

# Default model
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2-0.5B-Instruct}"
MODEL_DIR="./models/qwen2-onnx"
ONNX_PATH="$MODEL_DIR/model.onnx"

# Check if model exists
if [ ! -f "$ONNX_PATH" ]; then
    echo "ONNX model not found. Exporting $MODEL_NAME..."
    echo ""

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not found"
        echo "Please install Python 3.8+ to export the model"
        exit 1
    fi

    # Install dependencies
    echo "Installing Python dependencies..."
    pip3 install -q torch transformers onnx 2>/dev/null || {
        echo "⚠️  Warning: Could not install Python packages quietly"
        pip3 install torch transformers onnx
    }

    # Export to ONNX
    echo ""
    echo "Exporting model to ONNX (this may take 5-10 minutes)..."
    echo "Model: $MODEL_NAME"
    echo "Output: $MODEL_DIR"
    echo ""

    python3 scripts/export_to_onnx.py \
        --model "$MODEL_NAME" \
        --output "$MODEL_DIR" \
        --max-length 512 || {
        echo "❌ Model export failed"
        exit 1
    }

    # Create config
    cat > "$MODEL_DIR/nano_config.json" <<EOF
{
  "model_path": "$MODEL_DIR/model.onnx",
  "tokenizer_path": "$MODEL_DIR",
  "vocab_size": $(jq -r '.vocab_size // 151936' "$MODEL_DIR/model_info.json" 2>/dev/null || echo "151936"),
  "eos_token_id": $(jq -r '.eos_token_id // 151643' "$MODEL_DIR/model_info.json" 2>/dev/null || echo "151643"),
  "pad_token_id": $(jq -r '.pad_token_id // 151643' "$MODEL_DIR/model_info.json" 2>/dev/null || echo "151643")
}
EOF

    echo ""
fi

echo "✓ ONNX model found: $ONNX_PATH"
echo ""

# Build Go program
echo "Building Go program..."
go build -o bin/onnx_test ./purego/example_onnx || {
    echo "❌ Build failed"
    exit 1
}
echo "✓ Build complete"
echo ""

# Get prompt from args or use default
if [ $# -eq 0 ]; then
    PROMPT="What is the capital of France?"
else
    PROMPT="$*"
fi

# Run
echo "========================================="
echo "Running ONNX Runtime Inference"
echo "========================================="
echo ""

export MODEL_CONFIG="$MODEL_DIR/nano_config.json"

./bin/onnx_test "$PROMPT"

echo ""
echo "========================================="
echo "💡 Tips:"
echo "  • ONNX = Good speed + single binary"
echo "  • Speed: ~20-40 tok/s (CPU)"
echo "  • Perfect for production deployment"
echo ""
echo "Try another prompt:"
echo "  $0 \"Your question here\""
echo ""
echo "Use different model:"
echo "  MODEL_NAME='TinyLlama/TinyLlama-1.1B-Chat-v1.0' $0"
echo "========================================="
