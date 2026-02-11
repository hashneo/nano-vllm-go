#!/usr/bin/env python3
"""
Helper script to tokenize text using HuggingFace transformers.
Usage: python scripts/tokenize.py <model_dir> <text>
"""
import sys
from transformers import AutoTokenizer

if len(sys.argv) < 3:
    print("Usage: python scripts/tokenize.py <model_dir> <text>", file=sys.stderr)
    sys.exit(1)

model_dir = sys.argv[1]
text = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokens = tokenizer.encode(text, add_special_tokens=False)

# Output as comma-separated list for easy parsing in Go
print(','.join(map(str, tokens)))
