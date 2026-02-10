#!/bin/bash
# Script to download models compatible with nano-vllm-go

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║        Nano-vLLM-Go Compatible Model Downloader              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Function to download a model
download_model() {
    local model_name=$1
    local output_dir=$2
    local extra_flags=$3

    echo "Downloading: $model_name"
    echo "Output: $output_dir"
    echo ""

    python3 scripts/download_model.py --model "$model_name" --output "$output_dir" $extra_flags

    echo ""
    echo "✓ $model_name downloaded successfully!"
    echo ""
}

# Show menu
echo "Compatible Models:"
echo ""
echo "GPT-2 Models (Pure Go Implementation - Fully Working):"
echo "  1) GPT-2 Small (124M)    - Fast, ~500MB, 1K context"
echo "  2) GPT-2 Medium (355M)   - Balanced, ~1.4GB, 1K context"
echo "  3) GPT-2 Large (774M)    - Better quality, ~3GB, 1K context"
echo "  4) GPT-2 XL (1.5B)       - Best quality, ~6GB, 1K context"
echo ""
echo "Granite Models (Hybrid Attention+Mamba2 - Experimental):"
echo "  5) Granite 4 Nano 350M   - Small hybrid, ~1.3GB, 32K context"
echo "  6) Granite 4 Nano 1B     - Medium hybrid, ~3.8GB, 128K context"
echo ""
echo "  0) Exit"
echo ""
read -p "Select a model to download (1-6, or 0 to exit): " choice

case $choice in
    1)
        download_model "gpt2" "./models/gpt2-small"
        echo "Test with: ./bin/ask-gpt2 'The capital city of France is'"
        ;;
    2)
        download_model "gpt2-medium" "./models/gpt2-medium"
        echo "Note: Update cmd/ask-gpt2/main.go modelDir to use gpt2-medium"
        ;;
    3)
        download_model "gpt2-large" "./models/gpt2-large"
        echo "Note: Update cmd/ask-gpt2/main.go modelDir to use gpt2-large"
        ;;
    4)
        download_model "gpt2-xl" "./models/gpt2-xl"
        echo "Note: Update cmd/ask-gpt2/main.go modelDir to use gpt2-xl"
        ;;
    5)
        download_model "ibm-granite/granite-4.0-h-350m" "./models/granite-350m"
        echo "Test with: ./bin/generic-runner"
        ;;
    6)
        download_model "ibm-granite/granite-4.0-h-1b" "./models/granite-1b" "--fp16"
        echo "Test with: MODEL_DIR=./models/granite-1b ./bin/generic-runner"
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice: $choice"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Model ready! Build and run:"
echo "  make all"
echo "  ./demo_capitals.sh"
echo ""
