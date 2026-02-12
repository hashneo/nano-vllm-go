# HuggingFace Transformers Falcon Multi-Query Attention Implementation

## Source
**Repository**: https://github.com/huggingface/transformers
**File**: `src/transformers/models/falcon/modeling_falcon.py`
**Date Retrieved**: 2026-02-12

---

## 1. FalconAttention Class - Complete Implementation

### Initialization

```python
class FalconAttention(nn.Module):
    def __init__(self, config: FalconConfig, layer_idx=None):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.max_position_embeddings = config.max_position_embeddings

        self.is_causal = True
        self.layer_idx = layer_idx

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = self.inv_norm_factor

        # QKV output dimension varies by architecture
        if config.new_decoder_architecture:
            qkv_out_dim = (config.num_kv_heads * 2 + config.num_attention_heads) * self.head_dim
        elif config.multi_query:
            qkv_out_dim = self.hidden_size + 2 * self.head_dim
        else:
            qkv_out_dim = 3 * self.hidden_size

        self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
        self.new_decoder_architecture = config.new_decoder_architecture
        self.multi_query = config.multi_query
        self.dense = FalconLinear(self.hidden_size, self.hidden_size, bias=config.bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.num_kv_heads = config.num_kv_heads if (self.new_decoder_architecture or not self.multi_query) else 1
```

### Forward Pass

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    alibi: torch.Tensor | None,
    attention_mask: torch.Tensor,
    position_ids: torch.LongTensor | None = None,
    layer_past: Cache | None = None,
    use_cache: bool = False,
    output_attentions: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    # Step 1: Fused QKV projection
    fused_qkv = self.query_key_value(hidden_states)
    num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads

    # Step 2: Split into Q, K, V
    (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

    batch_size, query_length, _, _ = query_layer.shape

    # Step 3: Reshape to attention dimensions
    query_layer = query_layer.transpose(1, 2).reshape(batch_size, self.num_heads, query_length, self.head_dim)
    key_layer = key_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)
    value_layer = value_layer.transpose(1, 2).reshape(batch_size, num_kv_heads, query_length, self.head_dim)

    # Step 4: Apply rotary position embeddings (if not using alibi)
    if alibi is None:
        cos, sin = position_embeddings
        query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

    # Step 5: Update KV cache if needed
    if layer_past is not None:
        cache_kwargs = {"cache_position": cache_position}
        if alibi is None:
            cache_kwargs.update({"sin": sin, "cos": cos})
        key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

    kv_length = key_layer.shape[-2]

    # Step 6: Compute attention (two paths: with/without alibi)
    if alibi is None:
        # RoPE-based attention
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            is_causal = self.is_causal and attention_mask is None and query_length > 1
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
            attention_scores = None
        else:
            # Manual attention computation
            attention_scores = query_layer @ key_layer.transpose(-1, -2)
            attention_scores /= math.sqrt(self.head_dim)
            attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
            attn_output = attention_scores @ value_layer

        attn_output = attn_output.view(batch_size, self.num_heads, query_length, self.head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
        attn_output = self.dense(attn_output)
        return attn_output, attention_scores

    else:
        # ALiBi-based attention
        if self.config._attn_implementation == "sdpa" and not output_attentions:
            is_causal = self.is_causal and attention_mask is None and query_length > 1
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=is_causal,
            )
            attention_probs = None
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(batch_size, query_length, self.num_heads * self.head_dim)
            attn_output = self.dense(attn_output)
        else:
            # Manual attention computation with alibi
            matmul_result = query_layer @ key_layer.transpose(-1, -2)
            attention_scores = matmul_result.view(batch_size, self.num_heads, query_length, kv_length)

            # Convert to float32 for numerical stability
            input_dtype = attention_scores.dtype
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                attention_scores = attention_scores.to(torch.float32)

            # Add alibi bias BEFORE scaling
            attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)
            # Apply scaling
            attention_logits *= self.inv_norm_factor
            # Apply softmax with mask
            attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
            attention_probs = self.attention_dropout(attention_probs)

            attention_probs_reshaped = attention_probs.view(batch_size, self.num_heads, query_length, kv_length)
            attn_output = (attention_probs_reshaped @ value_layer).flatten(0, 1)
            attn_output = self._merge_heads(attn_output)
            attn_output = self.dense(attn_output)

        return attn_output, attention_probs
```

---

## 2. Query_Key_Value Projection Split

### The `_split_heads` Method

```python
def _split_heads(self, fused_qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim), results share same memory storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim]
        key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    if self.new_decoder_architecture:
        # New architecture with grouped query attention
        batch, seq_len, _ = fused_qkv.shape
        qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
        query = qkv[:, :, :, :-2]
        key = qkv[:, :, :, [-2]]
        value = qkv[:, :, :, [-1]]
        key = torch.broadcast_to(key, query.shape)
        value = torch.broadcast_to(value, query.shape)

        query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
        return query, key, value

    elif not self.multi_query:
        # Standard multi-head attention
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]

    else:
        # Multi-query attention (single KV head)
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads + 2, self.head_dim)
        return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]
```

### Reshape Operations Explained

#### For Multi-Query Attention (config.multi_query=True):

1. **Input shape**: `[batch, seq_len, hidden_size + 2 * head_dim]`
2. **Reshape to**: `[batch, seq_len, num_heads + 2, head_dim]`
3. **Split**:
   - Query: `[..., :-2, :]` → First `num_heads` slices → `[batch, seq_len, num_heads, head_dim]`
   - Key: `[..., [-2], :]` → Second-to-last slice → `[batch, seq_len, 1, head_dim]`
   - Value: `[..., [-1], :]` → Last slice → `[batch, seq_len, 1, head_dim]`

**Key observation**: K and V have shape `[..., 1, head_dim]` (single head), while Q has `[..., num_heads, head_dim]`

#### For Standard Multi-Head Attention (config.multi_query=False):

1. **Input shape**: `[batch, seq_len, 3 * hidden_size]`
2. **Reshape to**: `[batch, seq_len, num_heads, 3, head_dim]`
3. **Split** along dimension 3:
   - Query: `[..., 0, :]` → `[batch, seq_len, num_heads, head_dim]`
   - Key: `[..., 1, :]` → `[batch, seq_len, num_heads, head_dim]`
   - Value: `[..., 2, :]` → `[batch, seq_len, num_heads, head_dim]`

---

## 3. Attention Computation with Scaling

### Standard Scaling (1/sqrt(d_k))

For **RoPE-based attention** (alibi=None):

```python
attention_scores = query_layer @ key_layer.transpose(-1, -2)
attention_scores /= math.sqrt(self.head_dim)  # Standard 1/sqrt(d_k) scaling
attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
attn_output = attention_scores @ value_layer
```

### ALiBi-based Scaling (Different Order!)

For **ALiBi-based attention**:

```python
# Step 1: Compute raw attention scores
attention_scores = query_layer @ key_layer.transpose(-1, -2)

# Step 2: Add alibi bias BEFORE scaling
attention_logits = attention_scores + alibi.view(batch_size, self.num_heads, 1, -1)

# Step 3: Apply scaling (inv_norm_factor = 1.0 / math.sqrt(head_dim))
attention_logits *= self.inv_norm_factor

# Step 4: Apply softmax with mask
attention_probs = F.softmax(attention_logits + attention_mask, dim=-1, dtype=hidden_states.dtype)
```

**Critical difference**: ALiBi bias is added BEFORE scaling, not after. This is the correct order for ALiBi.

### Numerical Stability

```python
# Convert to float32 for mixed precision models
input_dtype = attention_scores.dtype
if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
    attention_scores = attention_scores.to(torch.float32)
```

---

## 4. KV Head Broadcasting to Query Heads

### In Multi-Query Attention Mode

After splitting, we have:
- Query: `[batch, num_heads, seq_len, head_dim]` (e.g., 32 heads)
- Key: `[batch, 1, seq_len, head_dim]` (single head)
- Value: `[batch, 1, seq_len, head_dim]` (single head)

**Broadcasting happens implicitly during attention computation**:

```python
# Matrix multiplication with broadcasting
attention_scores = query_layer @ key_layer.transpose(-1, -2)
# Shape: [batch, num_heads, seq_len, head_dim] @ [batch, 1, head_dim, seq_len]
# PyTorch broadcasts the "1" dimension to match num_heads
# Result: [batch, num_heads, seq_len, seq_len]

# Same for value projection
attn_output = attention_scores @ value_layer
# Shape: [batch, num_heads, seq_len, seq_len] @ [batch, 1, seq_len, head_dim]
# Broadcasting: [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
# Result: [batch, num_heads, seq_len, head_dim]
```

### In New Decoder Architecture

For grouped query attention:

```python
qkv = fused_qkv.view(batch, seq_len, -1, self.num_heads // self.num_kv_heads + 2, self.head_dim)
query = qkv[:, :, :, :-2]  # [batch, seq_len, num_kv_heads, num_queries_per_kv, head_dim]
key = qkv[:, :, :, [-2]]    # [batch, seq_len, num_kv_heads, 1, head_dim]
value = qkv[:, :, :, [-1]]  # [batch, seq_len, num_kv_heads, 1, head_dim]

# Explicit broadcasting
key = torch.broadcast_to(key, query.shape)
value = torch.broadcast_to(value, query.shape)

# Flatten back to standard shape
query, key, value = [x.flatten(2, 3) for x in (query, key, value)]
```

---

## 5. Causal Masking

### Causal Mask Application

The causal mask is applied in the softmax computation:

```python
# Mask is added to attention scores before softmax
attention_scores = F.softmax(attention_scores + attention_mask, dim=-1, dtype=hidden_states.dtype)
```

**Note**: The `attention_mask` is prepared upstream and includes:
- Causal mask (lower triangular)
- Padding mask (if applicable)
- ALiBi bias (if enabled, merged into mask)

### SDPA Path

When using PyTorch's optimized SDPA:

```python
is_causal = self.is_causal and attention_mask is None and query_length > 1
attn_output = torch.nn.functional.scaled_dot_product_attention(
    query_layer,
    key_layer,
    value_layer,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=is_causal,
)
```

The `is_causal=True` flag enables efficient causal masking inside SDPA.

---

## 6. Bias Handling

### FalconLinear Implementation

```python
class FalconLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_states = input @ self.weight.T
        if self.bias is None:
            return hidden_states
        return hidden_states + self.bias
```

**Key point**: Bias is added AFTER the matrix multiplication in a separate operation. This is intentional to preserve training characteristics where quantization to bfloat16 occurred between these operations.

### In Query_Key_Value Projection

```python
self.query_key_value = FalconLinear(self.hidden_size, qkv_out_dim, bias=config.bias)
```

Bias is added to the **fused QKV projection** before splitting:

1. `hidden_states @ weight.T` → produces fused QKV
2. `+ bias` → adds bias to entire fused output
3. `_split_heads()` → splits into Q, K, V

**Therefore: Bias is added BEFORE the split into Q, K, V**

---

## 7. Known Bugs and Fixes

### Bug #1: ALiBi Ignored with SDPA (Issue #29067)

**Problem**: When `_use_sdpa==True`, the ALiBi positional bias was being ignored in the attention computation.

**Root Cause**: The code had separate paths for SDPA and manual attention, and the SDPA path didn't include ALiBi application.

**Resolution**: ALiBi is now properly integrated into the attention mask at the top-level forward pass (lines 1104-1124 in modeling_falcon.py) before reaching the attention module, ensuring both SDPA and manual paths receive properly prepared masks.

### Bug #2: Different Outputs with output_attentions=True (Issue #29946)

**Problem**: The model produced different outputs when `output_attentions=True` vs `output_attentions=False`.

**Root Cause**: ALiBi bias was not being applied consistently between the two code paths.

**Fix (PR #30123)**: Ensured uniform ALiBi handling regardless of the `output_attentions` flag:

```python
# Proper approach: Handle ALiBi uniformly
alibi = 0.0 if alibi is None else alibi
attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attention_mask + (alibi * inv_norm_factor),
    ...
)
```

### Bug #3: Flash Attention 2.0 Support (Issues #28704, #26442, #26443)

**Problem**: Falcon models didn't support Flash Attention 2.0 initially.

**Resolution**: Support was added through architecture updates to make Falcon compatible with Flash Attention 2.0 optimizations.

---

## 8. Architecture Variations

Falcon supports three attention architectures:

### 1. Multi-Query Attention (original Falcon-7B/40B)
- **Config**: `multi_query=True`
- **QKV size**: `hidden_size + 2 * head_dim`
- **Characteristics**: Single KV head, all query heads share same K/V
- **Broadcasting**: Implicit via PyTorch broadcasting during matmul

### 2. Standard Multi-Head Attention
- **Config**: `multi_query=False`, `new_decoder_architecture=False`
- **QKV size**: `3 * hidden_size`
- **Characteristics**: Every head has its own Q, K, V
- **Broadcasting**: No broadcasting needed

### 3. New Decoder Architecture (Grouped Query Attention)
- **Config**: `new_decoder_architecture=True`
- **QKV size**: `(num_kv_heads * 2 + num_attention_heads) * head_dim`
- **Characteristics**: Multiple KV heads, each serving a group of query heads
- **Broadcasting**: Explicit via `torch.broadcast_to()`

---

## 9. Key Takeaways for Implementation

1. **QKV Projection**: Single fused projection with bias added before splitting
2. **Scaling Order**: For RoPE: scale before softmax. For ALiBi: add bias, then scale, then softmax
3. **KV Broadcasting**: Implicit in original multi-query, explicit in new decoder architecture
4. **Numerical Stability**: Convert to float32 for mixed precision models
5. **Causal Masking**: Applied via mask addition before softmax
6. **FalconLinear**: Separated matmul and bias for quantization compatibility
7. **Two Positional Encoding Modes**: Either RoPE (rotary) or ALiBi (linear bias), never both

---

## References

- Source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/modeling_falcon.py
- Issue #29067: ALiBi SDPA bug
- Issue #29946: output_attentions inconsistency
- PR #30123: Fix for ALiBi handling consistency
