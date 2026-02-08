#!/bin/bash
# Quick ONNX setup script for nano-vllm-go

set -e

echo "========================================="
echo "Nano-vLLM-Go - ONNX Quick Setup"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install -q torch transformers onnx || {
    echo "❌ Failed to install Python dependencies"
    exit 1
}
echo "✓ Python dependencies installed"

# Export model
echo ""
echo "Exporting model to ONNX (this may take a few minutes)..."
MODEL="${1:-Qwen/Qwen2-0.5B-Instruct}"
OUTPUT="${2:-./models/qwen2-onnx}"

echo "  Model: $MODEL"
echo "  Output: $OUTPUT"
echo ""

python3 scripts/export_to_onnx.py \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max-length 512 || {
    echo "❌ Model export failed"
    exit 1
}

echo ""
echo "✓ Model exported successfully!"

# Build Go program
echo ""
echo "Building Go program..."
go build -o bin/onnx_test ./purego/example_onnx || {
    echo "❌ Build failed"
    exit 1
}
echo "✓ Build complete"

# Create config file
echo ""
echo "Creating config file..."
cat > "$OUTPUT/nano_config.json" <<EOF
{
  "model_path": "$OUTPUT/model.onnx",
  "tokenizer_path": "$OUTPUT",
  "vocab_size": $(jq -r '.vocab_size // 32000' "$OUTPUT/model_info.json" 2>/dev/null || echo "32000"),
  "eos_token_id": $(jq -r '.eos_token_id // 151643' "$OUTPUT/model_info.json" 2>/dev/null || echo "151643"),
  "pad_token_id": $(jq -r '.pad_token_id // 0' "$OUTPUT/model_info.json" 2>/dev/null || echo "0")
}
EOF
echo "✓ Config created"

echo ""
echo "========================================="
echo "Setup complete! 🎉"
echo "========================================="
echo ""
echo "Test it with:"
echo "  export MODEL_CONFIG=$OUTPUT/nano_config.json"
echo "  ./bin/onnx_test \"What is the capital of France?\""
echo ""
echo "Or just run:"
echo "  MODEL_CONFIG=$OUTPUT/nano_config.json ./bin/onnx_test \"Your question?\""
echo ""
