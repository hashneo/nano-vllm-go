#!/usr/bin/env python3
"""
Export HuggingFace model to ONNX format for use with nano-vllm-go
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json


def export_to_onnx(model_name, output_dir, max_length=512):
    """Export model to ONNX format"""
    print(f"Loading model: {model_name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    print("Saving tokenizer...")
    tokenizer.save_pretrained(output_path)

    # Create dummy input
    print("Creating ONNX export...")
    dummy_input = torch.randint(0, tokenizer.vocab_size, (1, max_length))

    # Export to ONNX
    onnx_path = output_path / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size", 1: "sequence"}
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Save model info
    model_info = {
        "model_name": model_name,
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
        "pad_token_id": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None,
        "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else None,
        "num_layers": model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else None,
        "num_heads": model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else None,
    }

    with open(output_path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"✓ Model exported to: {output_path}")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Tokenizer: {output_path}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  EOS token: {tokenizer.eos_token_id}")


def main():
    parser = argparse.ArgumentParser(description="Export HuggingFace model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/qwen2-onnx",
        help="Output directory"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    args = parser.parse_args()

    try:
        export_to_onnx(args.model, args.output, args.max_length)
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
