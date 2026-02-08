#!/bin/bash
# Complete setup script for testing with a real model

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Nano-vLLM-Go Real Model Setup                            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
MODEL_NAME="${1:-Qwen/Qwen2-0.5B-Instruct}"
OUTPUT_DIR="./models"
ONNX_DIR="$OUTPUT_DIR/onnx"

echo "Model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"

# Create virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q torch transformers optimum[exporters] onnx onnxruntime

echo "✓ Dependencies installed"

# Download model
echo ""
echo "Downloading model: $MODEL_NAME"
mkdir -p "$OUTPUT_DIR"

python3 << EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "$MODEL_NAME"
output_dir = "$OUTPUT_DIR"

print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"Saving to {output_dir}...")
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)

print(f"✓ Model saved")
print(f"  - Vocab size: {tokenizer.vocab_size}")
print(f"  - EOS token: {tokenizer.eos_token_id}")
print(f"  - Model type: {model.config.model_type}")
EOF

# Convert to ONNX
echo ""
echo "Converting to ONNX format..."
mkdir -p "$ONNX_DIR"

python3 << EOF
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "$OUTPUT_DIR"
onnx_dir = "$ONNX_DIR"

print("Loading model for export...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create example input
example_text = "Hello"
input_ids = tokenizer.encode(example_text, return_tensors="pt")

print(f"Example input shape: {input_ids.shape}")
print(f"Tracing model...")

# Export to ONNX
with torch.no_grad():
    torch.onnx.export(
        model,
        input_ids,
        f"{onnx_dir}/model.onnx",
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True
    )

print(f"✓ ONNX model exported to {onnx_dir}/model.onnx")

# Save tokenizer
tokenizer.save_pretrained(onnx_dir)
print(f"✓ Tokenizer saved to {onnx_dir}/")
EOF

# Create config file
echo ""
echo "Creating model config..."
python3 << EOF
from transformers import AutoTokenizer
import json

tokenizer = AutoTokenizer.from_pretrained("$ONNX_DIR")

config = {
    "vocab_size": tokenizer.vocab_size,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
    "model_path": "$ONNX_DIR/model.onnx",
    "tokenizer_path": "$ONNX_DIR"
}

with open("$ONNX_DIR/nano_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✓ Config saved to $ONNX_DIR/nano_config.json")
print("")
print("Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
EOF

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ SETUP COMPLETE!                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model files created in: $ONNX_DIR/"
echo ""
echo "Next steps:"
echo "1. Install ONNX Runtime (if not already installed)"
echo "2. Build the example: make build-onnx"
echo "3. Test with a question: ./bin/onnx_example"
echo ""
echo "Or use the test script:"
echo "  ./scripts/test_real_model.sh"
echo ""
