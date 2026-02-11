# Tokenizer Limitations & TODO

## Current Status

The `UniversalTokenizer` has limited BPE (Byte-Pair Encoding) support:
- ✅ Loads vocabulary and special tokens correctly
- ✅ Handles special tokens like `<|begin_of_text|>`, `<|eot_id|>`
- ✅ Decoding works correctly (converts Ġ to spaces)
- ⚠️ Encoding does NOT apply BPE merge rules (280K+ rules for Llama 3.2)
- ⚠️ Falls back to word-level and character-level tokenization

**Working Solution:** `ask-llama` binary now automatically uses Python tokenizer helper for accurate encoding.

## Impact

For Llama 3.2 chat prompt:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is the capital of Germany?<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

- **Correct tokenization**: 17 tokens
- **Current UniversalTokenizer**: 25 tokens (incorrect for some words)

The model still works but may produce sub-optimal results with incorrect tokenization.

## Current Solution

### For End Users

The `ask-llama` binary automatically uses Python tokenizer integration:

```bash
# Just run - tokenization happens automatically
./bin/ask-llama "What is the capital of France?"
# Output: The capital of France is Paris.
```

No manual tokenization needed!

### For Developers

Manual tokenization with Python helper:

```bash
# Get tokens as comma-separated list
python3 scripts/encode_text.py models/llama-3.2-1b-instruct "Your text here"

# Example output:
# 128000,128006,882,128007,271,3923,374,279,6864,315,10057,30,128009
```

The `ask-llama` binary calls this script automatically via `encodeWithPython()` function.

## TODO

Implement proper BPE tokenization:

1. **Option A**: Integrate existing Go BPE library
   - Check: github.com/sugarme/tokenizer (HuggingFace tokenizers binding)
   - Pros: Full compatibility, maintained
   - Cons: External dependency

2. **Option B**: Implement BPE from scratch
   - Load merge rules from `tokenizer.json` (280K rules for Llama 3.2)
   - Apply merges iteratively to byte-encoded text
   - Pros: No external dependencies
   - Cons: Complex, performance-sensitive

3. **Option C**: CGo binding to HuggingFace tokenizers (Rust)
   - Use official tokenizers library
   - Pros: Perfect compatibility
   - Cons: Requires CGo, not pure Go

## Recommendation

For production: Use Option A (github.com/sugarme/tokenizer)
For testing: Use current workaround with Python tokenizer
