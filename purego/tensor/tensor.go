package tensor

import (
	"fmt"
	"math"
)

// Tensor represents a multi-dimensional array
type Tensor struct {
	Data  []float32
	Shape []int
}

// NewTensor creates a new tensor with given shape
func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: shape,
	}
}

// Size returns total number of elements
func (t *Tensor) Size() int {
	size := 1
	for _, dim := range t.Shape {
		size *= dim
	}
	return size
}

// At returns element at given indices
func (t *Tensor) At(indices ...int) float32 {
	idx := t.flatIndex(indices)
	return t.Data[idx]
}

// Set sets element at given indices
func (t *Tensor) Set(val float32, indices ...int) {
	idx := t.flatIndex(indices)
	t.Data[idx] = val
}

func (t *Tensor) flatIndex(indices []int) int {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("wrong number of indices: got %d, want %d", len(indices), len(t.Shape)))
	}
	idx := 0
	stride := 1
	for i := len(indices) - 1; i >= 0; i-- {
		idx += indices[i] * stride
		stride *= t.Shape[i]
	}
	return idx
}

// MatMul performs matrix multiplication: [m,k] x [k,n] -> [m,n]
func MatMul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("MatMul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		panic(fmt.Sprintf("incompatible shapes: [%d,%d] x [%d,%d]", a.Shape[0], a.Shape[1], b.Shape[0], b.Shape[1]))
	}

	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	result := NewTensor(m, n)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a.Data[i*k+p] * b.Data[p*n+j]
			}
			result.Data[i*n+j] = sum
		}
	}

	return result
}

// Add performs element-wise addition
func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensors must have same size")
	}
	result := NewTensor(a.Shape...)
	for i := range a.Data {
		result.Data[i] = a.Data[i] + b.Data[i]
	}
	return result
}

// Scale multiplies all elements by a scalar
func Scale(t *Tensor, factor float32) *Tensor {
	result := NewTensor(t.Shape...)
	for i := range t.Data {
		result.Data[i] = t.Data[i] * factor
	}
	return result
}

// Transpose swaps dimensions of a 2D tensor
func Transpose(t *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		panic("Transpose requires 2D tensor")
	}
	m, n := t.Shape[0], t.Shape[1]
	result := NewTensor(n, m)

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result.Data[j*m+i] = t.Data[i*n+j]
		}
	}
	return result
}

// Softmax applies softmax along the last dimension
func Softmax(t *Tensor) *Tensor {
	result := NewTensor(t.Shape...)

	// For each row (assuming 2D for simplicity)
	if len(t.Shape) == 2 {
		rows, cols := t.Shape[0], t.Shape[1]
		for i := 0; i < rows; i++ {
			// Find max for numerical stability
			maxVal := t.Data[i*cols]
			for j := 1; j < cols; j++ {
				if t.Data[i*cols+j] > maxVal {
					maxVal = t.Data[i*cols+j]
				}
			}

			// Compute exp and sum
			sum := float32(0)
			for j := 0; j < cols; j++ {
				val := float32(math.Exp(float64(t.Data[i*cols+j] - maxVal)))
				result.Data[i*cols+j] = val
				sum += val
			}

			// Normalize
			for j := 0; j < cols; j++ {
				result.Data[i*cols+j] /= sum
			}
		}
	} else {
		// Handle 1D
		maxVal := t.Data[0]
		for _, v := range t.Data {
			if v > maxVal {
				maxVal = v
			}
		}

		sum := float32(0)
		for i, v := range t.Data {
			val := float32(math.Exp(float64(v - maxVal)))
			result.Data[i] = val
			sum += val
		}

		for i := range result.Data {
			result.Data[i] /= sum
		}
	}

	return result
}

// GELU activation function
func GELU(t *Tensor) *Tensor {
	result := NewTensor(t.Shape...)
	for i, x := range t.Data {
		// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
		x3 := x * x * x
		inner := math.Sqrt(2.0/math.Pi) * float64(x+0.044715*x3)
		result.Data[i] = 0.5 * x * (1.0 + float32(math.Tanh(inner)))
	}
	return result
}

// LayerNorm applies layer normalization
func LayerNorm(t *Tensor, weight, bias *Tensor, eps float32) *Tensor {
	result := NewTensor(t.Shape...)

	// Determine if RMSNorm (no bias) or LayerNorm (with bias)
	isRMSNorm := (bias == nil)

	// Get the last dimension size (the dimension to normalize over)
	hiddenSize := t.Shape[len(t.Shape)-1]

	// Flatten to 2D: [batch*seq, hidden] or [batch, hidden]
	totalRows := 1
	for i := 0; i < len(t.Shape)-1; i++ {
		totalRows *= t.Shape[i]
	}

	for i := 0; i < totalRows; i++ {
		offset := i * hiddenSize

		if isRMSNorm {
			// RMSNorm: compute RMS (no mean subtraction)
			rms := float32(0)
			for j := 0; j < hiddenSize; j++ {
				val := t.Data[offset+j]
				rms += val * val
			}
			rms = float32(math.Sqrt(float64(rms/float32(hiddenSize) + eps)))

			// Normalize and scale
			for j := 0; j < hiddenSize; j++ {
				normalized := t.Data[offset+j] / rms
				result.Data[offset+j] = normalized * weight.Data[j]
			}
		} else {
			// LayerNorm: compute mean and variance
			mean := float32(0)
			for j := 0; j < hiddenSize; j++ {
				mean += t.Data[offset+j]
			}
			mean /= float32(hiddenSize)

			variance := float32(0)
			for j := 0; j < hiddenSize; j++ {
				diff := t.Data[offset+j] - mean
				variance += diff * diff
			}
			variance /= float32(hiddenSize)

			// Normalize
			std := float32(math.Sqrt(float64(variance + eps)))
			for j := 0; j < hiddenSize; j++ {
				normalized := (t.Data[offset+j] - mean) / std
				result.Data[offset+j] = normalized*weight.Data[j] + bias.Data[j]
			}
		}
	}

	return result
}

// Concatenate concatenates two tensors along a specified dimension
func Concatenate(t1, t2 *Tensor, dim int) *Tensor {
	// For now, only support sequence dimension (dim 2 for [batch, heads, seq, head_dim])
	if dim != 2 || len(t1.Shape) != 4 || len(t2.Shape) != 4 {
		panic("Concatenate only supports dim=2 for 4D tensors")
	}

	batch := t1.Shape[0]
	heads := t1.Shape[1]
	seq1 := t1.Shape[2]
	seq2 := t2.Shape[2]
	headDim := t1.Shape[3]

	// Create result tensor with combined sequence length
	result := NewTensor(batch, heads, seq1+seq2, headDim)

	// Copy data
	for b := 0; b < batch; b++ {
		for h := 0; h < heads; h++ {
			// Copy from t1
			for s := 0; s < seq1; s++ {
				for d := 0; d < headDim; d++ {
					srcIdx := ((b*heads+h)*seq1+s)*headDim + d
					dstIdx := ((b*heads+h)*(seq1+seq2)+s)*headDim + d
					result.Data[dstIdx] = t1.Data[srcIdx]
				}
			}
			// Copy from t2
			for s := 0; s < seq2; s++ {
				for d := 0; d < headDim; d++ {
					srcIdx := ((b*heads+h)*seq2+s)*headDim + d
					dstIdx := ((b*heads+h)*(seq1+seq2)+(seq1+s))*headDim + d
					result.Data[dstIdx] = t2.Data[srcIdx]
				}
			}
		}
	}

	return result
}

// Reshape returns a new tensor with different shape (same data)
func (t *Tensor) Reshape(shape ...int) *Tensor {
	newSize := 1
	for _, dim := range shape {
		newSize *= dim
	}
	if newSize != t.Size() {
		panic(fmt.Sprintf("cannot reshape: size mismatch %d vs %d", newSize, t.Size()))
	}
	result := &Tensor{
		Data:  t.Data,
		Shape: shape,
	}
	return result
}

// Slice extracts a slice along first dimension
func (t *Tensor) Slice(start, end int) *Tensor {
	if len(t.Shape) < 1 {
		panic("cannot slice scalar")
	}

	// Calculate size of one element along first dimension
	stride := 1
	for i := 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}

	newShape := make([]int, len(t.Shape))
	newShape[0] = end - start
	copy(newShape[1:], t.Shape[1:])

	startIdx := start * stride
	endIdx := end * stride

	result := &Tensor{
		Data:  t.Data[startIdx:endIdx],
		Shape: newShape,
	}
	return result
}
