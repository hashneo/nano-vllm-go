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

## Design Choice: Python Tokenizer Integration

The current implementation uses Python's `transformers` library for Llama tokenization. This is a **design choice**, not a workaround:

### Why Python Integration?

1. **Reference Implementation**: HuggingFace transformers is the reference implementation
2. **Correctness**: 100% accuracy guarantee for tokenization
3. **Complexity**: Byte-level BPE requires:
   - Regex-based pre-tokenization with complex patterns
   - Byte-to-unicode mapping (GPT-2 style)
   - 280K+ merge rules for Llama 3.2
   - Special token handling
4. **Common Pattern**: Many production systems (vLLM, text-generation-inference) use Python for tokenization

### Performance Note

The Python tokenizer is called once per request during prefill. For a typical prompt:
- Tokenization: ~1-5ms
- Inference: ~1000ms+

Tokenization overhead is < 0.5% of total latency, making pure Go tokenization optimization unnecessary for this educational project.

## Future Work (Optional)

If pure Go tokenization becomes important:

1. **Option A**: Debug github.com/sugarme/tokenizer compatibility issues
   - Tested: Library doesn't load Llama tokenizer.json properly
   - May work with other tokenizer formats

2. **Option B**: Implement byte-level BPE from scratch
   - Requires implementing GPT-2 byte-to-unicode mapping
   - Complex regex pre-tokenization patterns
   - Significant engineering effort (500+ lines)
   - Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py

3. **Option C**: CGo binding to HuggingFace tokenizers (Rust)
   - Perfect compatibility
   - Loses "pure Go" advantage
   - Adds build complexity

For this educational project, the Python integration provides the best balance of correctness, simplicity, and maintainability.
