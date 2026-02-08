#!/usr/bin/env python3
"""
Export a HuggingFace model to TorchScript for use with PyTorch implementation
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def export_model(model_name: str, output_path: str):
    """Export model to TorchScript"""
    print(f"Loading model: {model_name}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        trust_remote_code=True,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model loaded. Vocab size: {model.config.vocab_size}")

    # Create example input
    example_text = "Hello, world!"
    example_input = tokenizer.encode(example_text, return_tensors="pt")

    print(f"Example input shape: {example_input.shape}")

    # Trace the model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    # Save
    print(f"Saving to: {output_path}")
    torch.jit.save(traced_model, output_path)

    # Verify
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)

    with torch.no_grad():
        original_output = model(example_input)
        loaded_output = loaded_model(example_input)

        # Check outputs match
        if torch.allclose(original_output.logits, loaded_output.logits, rtol=1e-3):
            print("✓ Model exported successfully!")
        else:
            print("⚠ Warning: Outputs don't match perfectly")

    # Save tokenizer
    tokenizer_path = output_path.replace(".pt", "_tokenizer")
    print(f"Saving tokenizer to: {tokenizer_path}")
    tokenizer.save_pretrained(tokenizer_path)

    print("\nExport complete!")
    print(f"Model: {output_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Vocab size: {model.config.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")


def main():
    parser = argparse.ArgumentParser(description="Export HuggingFace model to TorchScript")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.pt",
        help="Output path for TorchScript model"
    )

    args = parser.parse_args()

    export_model(args.model, args.output)


if __name__ == "__main__":
    main()
