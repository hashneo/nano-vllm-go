#!/bin/bash
# Quick test script for real model inference

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         Testing Nano-vLLM-Go with Real Model                 ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if model exists
if [ ! -f "models/onnx/model.onnx" ]; then
    echo "❌ Model not found. Please run setup first:"
    echo "   ./scripts/setup_real_model.sh"
    echo ""
    exit 1
fi

echo "✓ Model found: models/onnx/model.onnx"
echo ""

# Build the example
echo "Building ONNX example..."
go build -o bin/onnx_example ./purego/example_onnx

echo "✓ Build complete"
echo ""

# Run with test questions
if [ $# -eq 0 ]; then
    echo "Running with default questions..."
    echo ""
    ./bin/onnx_example
else
    echo "Running with your question..."
    echo ""
    ./bin/onnx_example "$@"
fi
