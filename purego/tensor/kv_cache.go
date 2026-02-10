package tensor

// KVCache stores key-value tensors for efficient generation
type KVCache struct {
	Keys   []*Tensor // Per-layer key cache [batch, num_heads, seq_len, head_dim]
	Values []*Tensor // Per-layer value cache [batch, num_heads, seq_len, head_dim]
}

// NewKVCache creates a new KV cache for the model
func NewKVCache(numLayers int) *KVCache {
	return &KVCache{
		Keys:   make([]*Tensor, numLayers),
		Values: make([]*Tensor, numLayers),
	}
}

// GetLayer returns the KV cache for a specific layer
func (kv *KVCache) GetLayer(layerIdx int) (*Tensor, *Tensor) {
	if layerIdx < 0 || layerIdx >= len(kv.Keys) {
		return nil, nil
	}
	return kv.Keys[layerIdx], kv.Values[layerIdx]
}

// SetLayer sets the KV cache for a specific layer
func (kv *KVCache) SetLayer(layerIdx int, k, v *Tensor) {
	if layerIdx >= 0 && layerIdx < len(kv.Keys) {
		kv.Keys[layerIdx] = k
		kv.Values[layerIdx] = v
	}
}

// Clear resets the KV cache
func (kv *KVCache) Clear() {
	for i := range kv.Keys {
		kv.Keys[i] = nil
		kv.Values[i] = nil
	}
}
