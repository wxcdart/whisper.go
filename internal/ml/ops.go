package ml

import "context"

// MatMul computes C = A @ B.
// A: [M, K], B: [K, N] -> C: [M, N]. Parallelised over rows.
func MatMul(ctx context.Context, a, b Tensor) (Tensor, error) { panic("not implemented") }

// MatMulTransB computes C = A @ B^T.
func MatMulTransB(ctx context.Context, a, b Tensor) (Tensor, error) { panic("not implemented") }

// Add adds two tensors element-wise (broadcast over batch dim).
func Add(a, b Tensor) Tensor { panic("not implemented") }

// Mul multiplies two tensors element-wise.
func Mul(a, b Tensor) Tensor { panic("not implemented") }

// GELU applies the Gaussian Error Linear Unit activation.
func GELU(t Tensor) Tensor { panic("not implemented") }

// LayerNorm applies layer normalisation with weight and bias.
func LayerNorm(t, weight, bias Tensor, eps float32) Tensor { panic("not implemented") }

// Softmax applies softmax along the last axis.
func Softmax(t Tensor) Tensor { panic("not implemented") }

// Conv1D applies a 1-D convolution. weight: [outC, inC, kernel], bias: [outC].
func Conv1D(ctx context.Context, input, weight, bias Tensor, stride int) (Tensor, error) {
	panic("not implemented")
}

// Transpose permutes axes.
func Transpose(t Tensor, axes ...int) Tensor { panic("not implemented") }

// Concat concatenates tensors along the given axis.
func Concat(axis int, tensors ...Tensor) Tensor { panic("not implemented") }

// ScaledDotProductAttention computes Attention(Q,K,V) optionally with a causal mask.
// Returns output tensor and (if returnWeights) the attention weight matrix.
func ScaledDotProductAttention(ctx context.Context, q, k, v Tensor, causal bool, returnWeights bool) (out Tensor, weights Tensor, err error) {
	panic("not implemented")
}
