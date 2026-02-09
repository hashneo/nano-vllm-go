#!/usr/bin/env python3
"""
Universal model downloader for nano-vllm-go
Downloads any HuggingFace model and converts to safetensors format
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import save_file
from pathlib import Path
import json
import sys


def download_and_convert(model_name, output_dir, use_fp16=False):
    """Download any HuggingFace model and save in safetensors format"""
    print(f"Downloading {model_name}...")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load config first to check architecture
        print("\n1. Loading model configuration...")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        print(f"   Model type: {config.model_type}")

        # Load tokenizer
        print("\n2. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        print(f"   ✓ Tokenizer saved")

        # Load model
        print("\n3. Loading model...")
        if use_fp16:
            print("   Using float16 precision")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            print("   Using float32 precision")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        print(f"   ✓ Model loaded")
        print(f"   Config: {model.config}")

        # Save config
        config_path = output_path / "config.json"
        config.save_pretrained(output_path)
        print(f"\n4. ✓ Config saved to {config_path}")

        # Extract state dict
        print("\n5. Extracting model weights...")
        state_dict = model.state_dict()
        print(f"   Total tensors: {len(state_dict)}")

        # Convert to safetensors format
        safetensors_path = output_path / "model.safetensors"

        # Prepare tensors (convert to CPU and contiguous)
        print("\n6. Converting to safetensors format...")
        tensors = {}
        total_params = 0
        for name, param in state_dict.items():
            tensors[name] = param.cpu().contiguous()
            total_params += param.numel()

        # Save as safetensors
        save_file(tensors, str(safetensors_path))
        print(f"   ✓ Model saved to {safetensors_path}")
        print(f"   Total parameters: {total_params:,}")

        # Save model info (our custom format for easy loading)
        model_info = {
            "model_name": model_name,
            "model_type": config.model_type,
            "vocab_size": config.vocab_size,
            "hidden_size": getattr(config, 'hidden_size', getattr(config, 'n_embd', None)),
            "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None)),
            "num_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', None)),
            "max_seq_len": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', None)),
        }

        # Add special tokens
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            model_info["eos_token_id"] = tokenizer.eos_token_id
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            model_info["bos_token_id"] = tokenizer.bos_token_id
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            model_info["pad_token_id"] = tokenizer.pad_token_id

        info_path = output_path / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        print(f"\n7. ✓ Model info saved to {info_path}")

        # Print sample tensor names for debugging
        print("\n8. Sample tensor names:")
        for name in sorted(tensors.keys())[:15]:
            tensor = tensors[name]
            print(f"   {name}: {list(tensor.shape)}")
        if len(tensors) > 15:
            print(f"   ... ({len(tensors) - 15} more tensors)")

        print(f"\n{'='*70}")
        print(f"✓ SUCCESS! Model ready at {output_path}")
        print(f"{'='*70}")

        # Determine architecture for usage instructions
        architecture = "unknown"
        if "gpt2" in config.model_type.lower():
            architecture = "gpt2"
        elif "falcon" in config.model_type.lower():
            architecture = "falcon"
        elif "llama" in config.model_type.lower():
            architecture = "llama"
        elif "mistral" in config.model_type.lower():
            architecture = "mistral"
        elif "granite" in config.model_type.lower() or "granite" in model_name.lower():
            architecture = "granite"

        print(f"\nDetected architecture: {architecture}")
        print(f"\nUsage:")
        print(f"  export MODEL_DIR={output_path}")
        print(f"  ./bin/generic_test")
        print(f"\nOr with custom prompts:")
        print(f"  MODEL_DIR={output_path} ./bin/generic_test \"Your prompt here\"")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Download any HuggingFace model for nano-vllm-go",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download GPT-2 (small, 124M)
  ./scripts/download_model.py --model gpt2 --output ./models/gpt2

  # Download Falcon 7B
  ./scripts/download_model.py --model tiiuae/falcon-7b --output ./models/falcon-7b --fp16

  # Download Llama 2 7B (requires HF token)
  ./scripts/download_model.py --model meta-llama/Llama-2-7b-hf --output ./models/llama-7b --fp16

  # Download IBM Granite 4 Nano (350M)
  ./scripts/download_model.py --model ibm-granite/granite-4.0-h-350m --output ./models/granite-350m

  # Download IBM Granite 4 Nano (1B)
  ./scripts/download_model.py --model ibm-granite/granite-4.0-h-1b --output ./models/granite-1b --fp16

  # Download Mistral 7B
  ./scripts/download_model.py --model mistralai/Mistral-7B-v0.1 --output ./models/mistral-7b --fp16

Supported models:
  - GPT-2 family: gpt2, gpt2-medium, gpt2-large, gpt2-xl
  - Falcon: tiiuae/falcon-7b, tiiuae/falcon-40b
  - Llama 2: meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-13b-hf
  - Mistral: mistralai/Mistral-7B-v0.1
  - Granite: ibm-granite/granite-4.0-h-350m, ibm-granite/granite-4.0-h-1b
  - And many more HuggingFace models!
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'gpt2' or 'tiiuae/falcon-7b')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for the model"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 precision (recommended for large models to save memory)"
    )

    args = parser.parse_args()

    # Check if required packages are installed
    try:
        import transformers
        import safetensors
    except ImportError:
        print("❌ Error: Required packages not installed")
        print("\nPlease install:")
        print("  pip install transformers safetensors torch")
        return 1

    return download_and_convert(args.model, args.output, args.fp16)


if __name__ == "__main__":
    sys.exit(main())
