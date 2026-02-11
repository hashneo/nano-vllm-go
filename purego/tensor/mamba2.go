package tensor

import (
	"math"
)

// Mamba2Layer implements a Selective State-Space Model layer
// Based on "Transformers are SSMs" (Dao & Gu, 2024)
type Mamba2Layer struct {
	Config *ModelConfig

	// SSM parameters
	ALog      *Tensor // [num_heads, state_size] - log-space diagonal A matrix
	D         *Tensor // [expand * hidden] - skip connection
	Norm      *Tensor // [expand * hidden] - normalization scale
	DeltaBias *Tensor // [num_heads] - bias for time step

	// Projection weights
	InProj     *Tensor // [hidden, 2 * expand * hidden] - projects input to x and z
	ConvWeight *Tensor // [expand * hidden, 1, conv_kernel] - causal conv
	ConvBias   *Tensor // [expand * hidden] - conv bias
	XProj      *Tensor // [expand * hidden, dt_rank + 2 * n_groups * state_size] - projects to B, C, delta
	DtProj     *Tensor // [dt_rank, num_heads] - projects delta rank to heads
	OutProj    *Tensor // [expand * hidden, hidden] - output projection

	// Cache for inference
	ConvCache *Tensor // [batch, expand * hidden, conv_kernel]
	SSMState  *Tensor // [batch, num_heads, head_dim, state_size]

	// Config shortcuts
	hidden     int
	expand     int
	stateSize  int
	numHeads   int
	headDim    int
	nGroups    int
	convKernel int
	dtRank     int
	chunkSize  int
}

// NewMamba2Layer creates a new Mamba2 layer
func NewMamba2Layer(config *ModelConfig) *Mamba2Layer {
	layer := &Mamba2Layer{
		Config:     config,
		hidden:     config.Hidden,
		expand:     config.Mamba2Expand,
		stateSize:  config.Mamba2StateSize,
		numHeads:   config.Mamba2NumHeads,
		headDim:    config.Mamba2HeadDim,
		nGroups:    config.Mamba2NGroups,
		convKernel: config.Mamba2ConvKernel,
		dtRank:     config.Mamba2DtRank,
		chunkSize:  config.Mamba2ChunkSize,
	}

	// Use head dimension from config, or calculate if not set
	if layer.headDim == 0 {
		// Default: expand * hidden / num_heads
		layer.headDim = (layer.expand * layer.hidden) / layer.numHeads
	}

	// Calculate dt_rank if not set
	if layer.dtRank == 0 {
		layer.dtRank = int(math.Ceil(float64(layer.hidden) / 16.0))
	}

	return layer
}

// Forward performs the Mamba2 forward pass
// Input: x [batch, seq_len, hidden]
// Output: [batch, seq_len, hidden]
func (m *Mamba2Layer) Forward(x *Tensor) *Tensor {
	batch := x.Shape[0]
	seqLen := x.Shape[1]
	hidden := x.Shape[2]

	expandHidden := m.expand * m.hidden

	// 1. Input projection: x -> (gate/z, xBC, dt)
	// Granite Mamba2 splits in_proj output as:
	// - gate (z): intermediate_size (1536)
	// - xBC: conv_dim (1792) = intermediate_size + 2*n_groups*state_size
	// - dt: num_heads (48)
	// Total: 1536 + 1792 + 48 = 3376
	xFlat := x.Reshape(batch*seqLen, hidden)
	projected := MatMul(xFlat, Transpose(m.InProj)) // [batch*seq, in_proj_out]

	// Split into gate, xBC (goes to conv), dt
	gateSize := expandHidden                    // 1536 (intermediate_size)
	convDim := m.ConvWeight.Shape[0]            // 1792 (conv_dim = intermediate_size + 2*n_groups*state_size)
	dtSize := m.numHeads                        // 48

	// Split the projection: [gate | xBC | dt]
	gate := projected.SliceLastDim(0, gateSize)
	xBC := projected.SliceLastDim(gateSize, gateSize+convDim)
	dt := projected.SliceLastDim(gateSize+convDim, gateSize+convDim+dtSize)

	gate = gate.Reshape(batch, seqLen, gateSize)
	xBC = xBC.Reshape(batch, seqLen, convDim)
	dt = dt.Reshape(batch, seqLen, dtSize)

	// 2. Causal convolution on xBC (contains x, B, C together)
	xBCConv := m.causalConv1d(xBC)

	// 3. Activation (SiLU) on conv output
	xBCConv = SiLU(xBCConv)

	// 4. Split xBCConv into x, B, C
	// After conv: [intermediate_size | n_groups*state_size | n_groups*state_size]
	//           = [1536 | 128 | 128] = 1792
	xSSM := xBCConv.SliceLastDim(0, expandHidden)
	B := xBCConv.SliceLastDim(expandHidden, expandHidden+m.nGroups*m.stateSize)
	C := xBCConv.SliceLastDim(expandHidden+m.nGroups*m.stateSize, convDim)

	// Reshape B and C to [batch, seq, n_groups, state_size]
	B = B.Reshape(batch, seqLen, m.nGroups, m.stateSize)
	C = C.Reshape(batch, seqLen, m.nGroups, m.stateSize)

	// 5. Process delta: add bias and apply softplus
	var delta *Tensor = dt
	if m.DeltaBias != nil {
		for i := 0; i < batch*seqLen*m.numHeads; i++ {
			headIdx := i % m.numHeads
			delta.Data[i] += m.DeltaBias.Data[headIdx]
		}
	}
	delta = Softplus(delta)

	// 6. Selective scan (core SSM operation)
	y := m.selectiveScan(xSSM, B, C, delta, batch, seqLen)

	// 7. Apply RMSNormGated: combines gating and normalization
	// This is equivalent to PyTorch's GraniteMoeHybridRMSNormGated
	// Step 1: Apply gating (hidden_states * silu(gate))
	gateActivated := SiLU(gate)
	for i := 0; i < len(y.Data); i++ {
		yIdx := i
		gateIdx := (i / expandHidden) * gateSize + (i % expandHidden)
		y.Data[yIdx] *= gateActivated.Data[gateIdx]
	}

	// Step 2: RMS normalization per sequence position
	eps := float32(1e-5)
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			// Calculate variance
			var variance float32 = 0
			offset := b*seqLen*expandHidden + s*expandHidden
			for d := 0; d < expandHidden; d++ {
				val := y.Data[offset+d]
				variance += val * val
			}
			variance /= float32(expandHidden)

			// Apply RMS normalization
			rms := float32(1.0) / float32(math.Sqrt(float64(variance+eps)))
			for d := 0; d < expandHidden; d++ {
				y.Data[offset+d] *= rms
			}
		}
	}

	// Step 3: Apply learned weight (if exists)
	if m.Norm != nil {
		for i := 0; i < len(y.Data); i++ {
			y.Data[i] *= m.Norm.Data[i%expandHidden]
		}
	}

	// 9. Output projection
	// OutProj is [hidden, expand*hidden] in PyTorch format, need to transpose
	yFlat := y.Reshape(batch*seqLen, expandHidden)
	output := MatMul(yFlat, Transpose(m.OutProj))
	output = output.Reshape(batch, seqLen, m.hidden)

	return output
}

// causalConv1d performs causal 1D convolution with padding
// PyTorch conv1d uses padding=kernel_size-1, then slices back to original length
func (m *Mamba2Layer) causalConv1d(x *Tensor) *Tensor {
	batch := x.Shape[0]
	seqLen := x.Shape[1]
	channels := x.Shape[2]

	// PyTorch adds padding=kernel_size-1 on the left (past)
	padding := m.convKernel - 1
	paddedSeqLen := seqLen + padding

	// Create padded input (zero-padded on the left)
	paddedInput := NewTensor(batch, paddedSeqLen, channels)
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				// Copy input to padded position (offset by padding)
				srcIdx := b*seqLen*channels + t*channels + c
				dstIdx := b*paddedSeqLen*channels + (t+padding)*channels + c
				paddedInput.Data[dstIdx] = x.Data[srcIdx]
			}
		}
	}

	// Convolve on padded input
	paddedResult := NewTensor(batch, paddedSeqLen, channels)
	for b := 0; b < batch; b++ {
		for t := 0; t < paddedSeqLen; t++ {
			for c := 0; c < channels; c++ {
				sum := float32(0)

				// Apply convolution (PyTorch applies kernel in forward direction)
				for k := 0; k < m.convKernel; k++ {
					pos := t + k
					if pos >= 0 && pos < paddedSeqLen {
						xIdx := b*paddedSeqLen*channels + pos*channels + c
						wIdx := c*m.convKernel + k
						sum += paddedInput.Data[xIdx] * m.ConvWeight.Data[wIdx]
					}
				}

				// Add bias
				if m.ConvBias != nil {
					sum += m.ConvBias.Data[c]
				}

				outIdx := b*paddedSeqLen*channels + t*channels + c
				paddedResult.Data[outIdx] = sum
			}
		}
	}

	// Slice back to original sequence length
	// PyTorch slices [..., :seqLen] which takes first seqLen positions
	result := NewTensor(batch, seqLen, channels)
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				srcIdx := b*paddedSeqLen*channels + t*channels + c
				dstIdx := b*seqLen*channels + t*channels + c
				result.Data[dstIdx] = paddedResult.Data[srcIdx]
			}
		}
	}

	return result
}

// selectiveScan performs the core selective state-space scan
func (m *Mamba2Layer) selectiveScan(x, B, C, delta *Tensor, batch, seqLen int) *Tensor {
	// x: [batch, seq, expand*hidden]
	// B: [batch, seq, n_groups, state_size]
	// C: [batch, seq, n_groups, state_size]
	// delta: [batch, seq, num_heads]

	expandHidden := m.expand * m.hidden

	// Reshape x to [batch, seq, num_heads, head_dim]
	xReshaped := x.Reshape(batch, seqLen, m.numHeads, m.headDim)

	// Initialize or get cached state
	if m.SSMState == nil {
		m.SSMState = NewTensor(batch, m.numHeads, m.headDim, m.stateSize)
	}

	// Output tensor
	output := NewTensor(batch, seqLen, m.numHeads, m.headDim)

	// Process each timestep (can be chunked for efficiency)
	for t := 0; t < seqLen; t++ {
		for b := 0; b < batch; b++ {
			// Process each head
			for h := 0; h < m.numHeads; h++ {
				// Get delta for this timestep/batch/head
				deltaIdx := b*seqLen*m.numHeads + t*m.numHeads + h
				dt := delta.Data[deltaIdx]

				// Discretize A matrix using zero-order hold (ZOH)
				// In Mamba/Mamba2, A_log represents log(-A) where A is the continuous-time decay matrix
				// Discretization:
				//   A_bar = exp(A * dt) = exp(-exp(A_log) * dt)
				//   B_bar = (A_bar - 1) / A * B = (1 - exp(-exp(A_log)*dt)) / exp(A_log) * B
				// For simplicity, use Euler discretization: B_bar ≈ dt * B
				var ABar float32 = 1.0
				if m.ALog != nil && len(m.ALog.Data) > h {
					// A = -exp(A_log), so A_bar = exp(A * dt) = exp(-exp(A_log) * dt)
					A := -float32(math.Exp(float64(m.ALog.Data[h])))
					ABar = float32(math.Exp(float64(A * dt)))
				}

				// Get input for this timestep
				// u: [head_dim]
				u := make([]float32, m.headDim)
				for d := 0; d < m.headDim; d++ {
					uIdx := b*seqLen*m.numHeads*m.headDim + t*m.numHeads*m.headDim + h*m.headDim + d
					u[d] = xReshaped.Data[uIdx]
				}

				// Get B and C for this timestep
				// Map head to group
				groupIdx := h * m.nGroups / m.numHeads
				if groupIdx >= m.nGroups {
					groupIdx = m.nGroups - 1
				}

				// B_t, C_t: [state_size]
				Bt := make([]float32, m.stateSize)
				Ct := make([]float32, m.stateSize)
				for s := 0; s < m.stateSize; s++ {
					bIdx := b*seqLen*m.nGroups*m.stateSize + t*m.nGroups*m.stateSize + groupIdx*m.stateSize + s
					cIdx := bIdx
					Bt[s] = B.Data[bIdx]
					Ct[s] = C.Data[cIdx]
				}

				// Update state for each dimension of head_dim
				for d := 0; d < m.headDim; d++ {
					for s := 0; s < m.stateSize; s++ {
						stateIdx := b*m.numHeads*m.headDim*m.stateSize + h*m.headDim*m.stateSize + d*m.stateSize + s

						// State update: x[n+1] = A_bar * x[n] + dt * B * u[n]
						oldState := m.SSMState.Data[stateIdx]
						// ABar is now a scalar per head, broadcast to all states
						m.SSMState.Data[stateIdx] = ABar*oldState + dt*Bt[s]*u[d]
					}
				}

				// Compute output: y = C @ state + D * u
				for d := 0; d < m.headDim; d++ {
					sum := float32(0)
					for s := 0; s < m.stateSize; s++ {
						stateIdx := b*m.numHeads*m.headDim*m.stateSize + h*m.headDim*m.stateSize + d*m.stateSize + s
						sum += Ct[s] * m.SSMState.Data[stateIdx]
					}

					// Add skip connection (D * u)
					// In Granite, D is [num_heads], one value per head
					if m.D != nil && len(m.D.Data) > h {
						sum += m.D.Data[h] * u[d]
					}

					outIdx := b*seqLen*m.numHeads*m.headDim + t*m.numHeads*m.headDim + h*m.headDim + d
					output.Data[outIdx] = sum
				}
			}
		}
	}

	// Reshape back to [batch, seq, expand*hidden]
	return output.Reshape(batch, seqLen, expandHidden)
}

// ResetState clears the SSM state cache
func (m *Mamba2Layer) ResetState() {
	m.SSMState = nil
	m.ConvCache = nil
}

// SiLU activation (Sigmoid Linear Unit)
func SiLU(x *Tensor) *Tensor {
	result := NewTensor(x.Shape...)
	for i := 0; i < len(x.Data); i++ {
		sigmoid := 1.0 / (1.0 + math.Exp(-float64(x.Data[i])))
		result.Data[i] = x.Data[i] * float32(sigmoid)
	}
	return result
}

// Softplus activation (smooth ReLU)
func Softplus(x *Tensor) *Tensor {
	result := NewTensor(x.Shape...)
	for i := 0; i < len(x.Data); i++ {
		result.Data[i] = float32(math.Log(1.0 + math.Exp(float64(x.Data[i]))))
	}
	return result
}

// SliceLastDim slices the last dimension of a tensor
func (t *Tensor) SliceLastDim(start, end int) *Tensor {
	if len(t.Shape) == 0 {
		return t
	}

	lastDim := t.Shape[len(t.Shape)-1]
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[len(newShape)-1] = end - start

	result := NewTensor(newShape...)

	// Calculate strides
	totalBefore := 1
	for i := 0; i < len(t.Shape)-1; i++ {
		totalBefore *= t.Shape[i]
	}

	for i := 0; i < totalBefore; i++ {
		srcOffset := i * lastDim
		dstOffset := i * (end - start)
		copy(result.Data[dstOffset:dstOffset+(end-start)], t.Data[srcOffset+start:srcOffset+end])
	}

	return result
}

// SetConfig allows Mamba2Layer to access config
func (m *Mamba2Layer) SetConfig(config *ModelConfig) {
	m.Config = config
}

// Verify Mamba2Layer implements AttentionLayer interface
var _ AttentionLayer = (*Mamba2Layer)(nil)
