package tensor

import (
	"testing"
)

func TestSplitFalconQKV(t *testing.T) {
	// Test with simplified Falcon dimensions
	// Real Falcon-7B: 71 heads, 64 head_dim, 4544 hidden
	// Test: 3 heads, 4 head_dim, 12 hidden
	numHeads := 3
	headDim := 4
	hidden := numHeads * headDim // 12

	// Create a test QKV weight matrix
	// Shape: [hidden, (num_heads + 2) * head_dim] = [12, 20]
	// Layout per row: [Q0(4), Q1(4), Q2(4), K(4), V(4)]
	qkvData := make([]float32, hidden*(numHeads+2)*headDim)

	// Fill with identifiable patterns
	for row := 0; row < hidden; row++ {
		offset := row * (numHeads + 2) * headDim

		// Q heads: fill each head with unique value
		for h := 0; h < numHeads; h++ {
			for d := 0; d < headDim; d++ {
				qkvData[offset+h*headDim+d] = float32(10*row + h)
			}
		}

		// K: fill with 100*row
		for d := 0; d < headDim; d++ {
			qkvData[offset+numHeads*headDim+d] = float32(100 * row)
		}

		// V: fill with 1000*row
		for d := 0; d < headDim; d++ {
			qkvData[offset+(numHeads+1)*headDim+d] = float32(1000 * row)
		}
	}

	qkvWeight := &Tensor{
		Data:  qkvData,
		Shape: []int{hidden, (numHeads + 2) * headDim},
	}

	config := &ModelConfig{
		Hidden:   hidden,
		HeadDim:  headDim,
		NumHeads: numHeads,
	}

	// Test the split function
	Q, K, V := splitFalconQKV(qkvWeight, config)

	// Verify shapes
	if len(Q.Shape) != 2 || Q.Shape[0] != hidden || Q.Shape[1] != numHeads*headDim {
		t.Errorf("Q shape incorrect: got %v, want [%d, %d]", Q.Shape, hidden, numHeads*headDim)
	}
	if len(K.Shape) != 2 || K.Shape[0] != hidden || K.Shape[1] != headDim {
		t.Errorf("K shape incorrect: got %v, want [%d, %d]", K.Shape, hidden, headDim)
	}
	if len(V.Shape) != 2 || V.Shape[0] != hidden || V.Shape[1] != headDim {
		t.Errorf("V shape incorrect: got %v, want [%d, %d]", V.Shape, hidden, headDim)
	}

	// Verify Q values (check first few rows and heads)
	for row := 0; row < 3; row++ {
		for h := 0; h < numHeads; h++ {
			idx := row*numHeads*headDim + h*headDim
			val := Q.Data[idx]
			expected := float32(10*row + h)
			if val != expected {
				t.Errorf("Q[%d][head%d] = %.1f, want %.1f", row, h, val, expected)
			}
		}
	}

	// Verify K values
	for row := 0; row < 3; row++ {
		idx := row * headDim
		val := K.Data[idx]
		expected := float32(100 * row)
		if val != expected {
			t.Errorf("K[%d] = %.1f, want %.1f", row, val, expected)
		}
	}

	// Verify V values
	for row := 0; row < 3; row++ {
		idx := row * headDim
		val := V.Data[idx]
		expected := float32(1000 * row)
		if val != expected {
			t.Errorf("V[%d] = %.1f, want %.1f", row, val, expected)
		}
	}
}

func TestSplitFalconQKV_RealFalcon7BDimensions(t *testing.T) {
	// Test with actual Falcon-7B dimensions (but smaller data)
	numHeads := 71
	headDim := 64
	hidden := 4544

	// Create weight matrix with pattern
	qkvData := make([]float32, hidden*(numHeads+2)*headDim)

	// Just fill first row as a sanity check
	for h := 0; h < numHeads; h++ {
		for d := 0; d < headDim; d++ {
			qkvData[h*headDim+d] = float32(h) // Each head has its index
		}
	}
	// K
	for d := 0; d < headDim; d++ {
		qkvData[numHeads*headDim+d] = 999.0
	}
	// V
	for d := 0; d < headDim; d++ {
		qkvData[(numHeads+1)*headDim+d] = 888.0
	}

	qkvWeight := &Tensor{
		Data:  qkvData,
		Shape: []int{hidden, (numHeads + 2) * headDim},
	}

	config := &ModelConfig{
		Hidden:   hidden,
		HeadDim:  headDim,
		NumHeads: numHeads,
	}

	Q, K, V := splitFalconQKV(qkvWeight, config)

	// Verify shapes
	if Q.Shape[0] != hidden || Q.Shape[1] != numHeads*headDim {
		t.Errorf("Q shape incorrect: got %v, want [%d, %d]", Q.Shape, hidden, numHeads*headDim)
	}

	// Check first row values
	for h := 0; h < numHeads; h++ {
		val := Q.Data[h*headDim]
		expected := float32(h)
		if val != expected {
			t.Errorf("Q[0][head%d] = %.1f, want %.1f", h, val, expected)
		}
	}

	if K.Data[0] != 999.0 {
		t.Errorf("K[0] = %.1f, want 999.0", K.Data[0])
	}

	if V.Data[0] != 888.0 {
		t.Errorf("V[0] = %.1f, want 888.0", V.Data[0])
	}
}
