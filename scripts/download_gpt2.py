#!/usr/bin/env python3
"""
Download GPT-2 model and convert to safetensors format
"""

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors.torch import save_file
from pathlib import Path
import json


def download_and_convert(model_name="gpt2", output_dir="./models/gpt2"):
    """Download GPT-2 and save in safetensors format"""
    print(f"Downloading {model_name}...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    print(f"✓ Model loaded")
    print(f"  Config: {model.config}")

    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"✓ Tokenizer saved to {output_path}")

    # Extract state dict
    state_dict = model.state_dict()

    # Convert to safetensors format
    safetensors_path = output_path / "model.safetensors"

    # Prepare tensors (convert to CPU and contiguous)
    tensors = {}
    for name, param in state_dict.items():
        tensors[name] = param.cpu().contiguous()

    # Save as safetensors
    save_file(tensors, str(safetensors_path))
    print(f"✓ Model saved to {safetensors_path}")

    # Save model info
    model_info = {
        "model_name": model_name,
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.n_embd,
        "num_layers": model.config.n_layer,
        "num_heads": model.config.n_head,
        "max_seq_len": model.config.n_positions,
        "eos_token_id": tokenizer.eos_token_id,
    }

    info_path = output_path / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"✓ Model info saved to {info_path}")

    # Print tensor names for debugging
    print("\nTensor names in model:")
    for name in sorted(tensors.keys())[:10]:
        tensor = tensors[name]
        print(f"  {name}: {list(tensor.shape)}")
    print(f"  ... ({len(tensors)} total tensors)")

    print(f"\n✓ Done! Model ready at {output_path}")
    print(f"\nUsage:")
    print(f"  export MODEL_PATH={output_path}/model.safetensors")
    print(f"  export TOKENIZER_PATH={output_path}")
    print(f"  go run ./purego/example_native/main.go")


def main():
    parser = argparse.ArgumentParser(description="Download GPT-2 model")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model size",
    )
    parser.add_argument(
        "--output", type=str, default="./models/gpt2", help="Output directory"
    )

    args = parser.parse_args()

    try:
        download_and_convert(args.model, args.output)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
