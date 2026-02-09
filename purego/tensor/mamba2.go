package tensor

import (
	"math"
)

// Mamba2Layer implements a Selective State-Space Model layer
// Based on "Transformers are SSMs" (Dao & Gu, 2024)
type Mamba2Layer struct {
	Config *ModelConfig

	// SSM parameters
	ALog        *Tensor // [num_heads, state_size] - log-space diagonal A matrix
	D           *Tensor // [expand * hidden] - skip connection
	Norm        *Tensor // [expand * hidden] - normalization scale
	DeltaBias   *Tensor // [num_heads] - bias for time step

	// Projection weights
	InProj      *Tensor // [hidden, 2 * expand * hidden] - projects input to x and z
	ConvWeight  *Tensor // [expand * hidden, 1, conv_kernel] - causal conv
	ConvBias    *Tensor // [expand * hidden] - conv bias
	XProj       *Tensor // [expand * hidden, dt_rank + 2 * n_groups * state_size] - projects to B, C, delta
	DtProj      *Tensor // [dt_rank, num_heads] - projects delta rank to heads
	OutProj     *Tensor // [expand * hidden, hidden] - output projection

	// Cache for inference
	ConvCache   *Tensor // [batch, expand * hidden, conv_kernel]
	SSMState    *Tensor // [batch, num_heads, head_dim, state_size]

	// Config shortcuts
	hidden      int
	expand      int
	stateSize   int
	numHeads    int
	headDim     int
	nGroups     int
	convKernel  int
	dtRank      int
	chunkSize   int
}

// NewMamba2Layer creates a new Mamba2 layer
func NewMamba2Layer(config *ModelConfig) *Mamba2Layer {
	layer := &Mamba2Layer{
		Config:     config,
		hidden:     config.Hidden,
		expand:     config.Mamba2Expand,
		stateSize:  config.Mamba2StateSize,
		numHeads:   config.Mamba2NumHeads,
		nGroups:    config.Mamba2NGroups,
		convKernel: config.Mamba2ConvKernel,
		dtRank:     config.Mamba2DtRank,
		chunkSize:  config.Mamba2ChunkSize,
	}

	// Calculate head dimension
	layer.headDim = layer.stateSize / layer.numHeads

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

	// 1. Input projection: x -> (x, z)
	xz := MatMul(x.Reshape(batch*seqLen, hidden), m.InProj) // [batch*seq, 2*expand*hidden]
	expandHidden := m.expand * m.hidden

	// Split into x and z (for gating)
	xPart := xz.SliceLastDim(0, expandHidden)
	zPart := xz.SliceLastDim(expandHidden, 2*expandHidden)

	xPart = xPart.Reshape(batch, seqLen, expandHidden)
	zPart = zPart.Reshape(batch, seqLen, expandHidden)

	// 2. Causal convolution
	xConv := m.causalConv1d(xPart)

	// 3. Activation (SiLU)
	xConv = SiLU(xConv)

	// 4. SSM projection to B, C, delta
	xFlat := xConv.Reshape(batch*seqLen, expandHidden)
	xProj := MatMul(xFlat, m.XProj) // [batch*seq, dt_rank + 2*n_groups*state_size]
	xProj = xProj.Reshape(batch, seqLen, -1)

	// Split into delta_input, B, C
	dtRank := m.dtRank
	BCSize := 2 * m.nGroups * m.stateSize

	deltaInput := xProj.SliceLastDim(0, dtRank)
	BC := xProj.SliceLastDim(dtRank, dtRank+BCSize)

	// Split B and C
	B := BC.SliceLastDim(0, m.nGroups*m.stateSize)
	C := BC.SliceLastDim(m.nGroups*m.stateSize, BCSize)

	// Reshape B and C to [batch, seq, n_groups, state_size]
	B = B.Reshape(batch, seqLen, m.nGroups, m.stateSize)
	C = C.Reshape(batch, seqLen, m.nGroups, m.stateSize)

	// 5. Project delta from dt_rank to num_heads
	deltaFlat := deltaInput.Reshape(batch*seqLen, dtRank)
	delta := MatMul(deltaFlat, m.DtProj) // [batch*seq, num_heads]
	delta = delta.Reshape(batch, seqLen, m.numHeads)

	// Add bias and apply softplus to keep positive
	if m.DeltaBias != nil {
		for i := 0; i < batch*seqLen*m.numHeads; i++ {
			headIdx := i % m.numHeads
			delta.Data[i] += m.DeltaBias.Data[headIdx]
		}
	}
	delta = Softplus(delta)

	// 6. Selective scan (core SSM operation)
	y := m.selectiveScan(xConv, B, C, delta, batch, seqLen)

	// 7. Apply normalization
	if m.Norm != nil {
		for i := 0; i < len(y.Data); i++ {
			y.Data[i] *= m.Norm.Data[i%expandHidden]
		}
	}

	// 8. Apply gating with z
	zActivated := SiLU(zPart)
	for i := 0; i < len(y.Data); i++ {
		y.Data[i] *= zActivated.Data[i]
	}

	// 9. Output projection
	yFlat := y.Reshape(batch*seqLen, expandHidden)
	output := MatMul(yFlat, m.OutProj)
	output = output.Reshape(batch, seqLen, m.hidden)

	return output
}

// causalConv1d performs causal 1D convolution
func (m *Mamba2Layer) causalConv1d(x *Tensor) *Tensor {
	batch := x.Shape[0]
	seqLen := x.Shape[1]
	channels := x.Shape[2]

	result := NewTensor(batch, seqLen, channels)

	// Simple causal convolution
	for b := 0; b < batch; b++ {
		for t := 0; t < seqLen; t++ {
			for c := 0; c < channels; c++ {
				sum := float32(0)

				// Causal: only look at current and past
				for k := 0; k < m.convKernel; k++ {
					pos := t - k
					if pos >= 0 {
						// x[b, pos, c] * weight[c, 0, k]
						xIdx := b*seqLen*channels + pos*channels + c
						wIdx := c*m.convKernel + k
						sum += x.Data[xIdx] * m.ConvWeight.Data[wIdx]
					}
				}

				// Add bias
				if m.ConvBias != nil {
					sum += m.ConvBias.Data[c]
				}

				outIdx := b*seqLen*channels + t*channels + c
				result.Data[outIdx] = sum
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

				// Discretize A matrix: A_bar = exp(dt * A_log)
				// A is diagonal, stored in log space
				var ABar []float32
				if m.ALog != nil {
					ABar = make([]float32, m.stateSize)
					for s := 0; s < m.stateSize; s++ {
						aIdx := h*m.stateSize + s
						ABar[s] = float32(math.Exp(float64(dt * m.ALog.Data[aIdx])))
					}
				} else {
					ABar = make([]float32, m.stateSize)
					for s := 0; s < m.stateSize; s++ {
						ABar[s] = 1.0
					}
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
						m.SSMState.Data[stateIdx] = ABar[s]*oldState + dt*Bt[s]*u[d]
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
					if m.D != nil {
						dIdx := h*m.headDim + d
						sum += m.D.Data[dIdx] * u[d]
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
