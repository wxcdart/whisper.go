package model

import (
	"context"

	"github.com/whispergo/whisper.go/internal/ml"
)

// ComputeBackend abstracts the subset of tensor ops used by model inference.
// The default backend routes directly to internal/ml.
type ComputeBackend interface {
	Conv1D(ctx context.Context, x, w, b ml.Tensor, stride int) (ml.Tensor, error)
	GELU(x ml.Tensor) ml.Tensor
	GELUInPlace(x ml.Tensor)
	Add(a, b ml.Tensor) ml.Tensor
	LayerNorm(x, weight, bias ml.Tensor, eps float32) ml.Tensor
	LayerNormInto(dst, x, weight, bias ml.Tensor, eps float32) error
	Transpose(x ml.Tensor, axes ...int) ml.Tensor
	ScaledDotProductAttention(ctx context.Context, q, k, v ml.Tensor, causal, returnScores bool) (ml.Tensor, ml.Tensor, error)
	ScaledDotProductAttentionInto(ctx context.Context, q, k, v ml.Tensor, causal bool, out, scratch, scores ml.Tensor) error
	MatMulTransB(ctx context.Context, a, b ml.Tensor) (ml.Tensor, error)
	MatMulTransBInto(ctx context.Context, a, b, out ml.Tensor) error
	MatMulQuantTransBInto(ctx context.Context, a ml.Tensor, b ml.QuantizedMatrix, out ml.Tensor) error
	ShouldUseQuantMatMul(m, k, n int, qtype uint32) bool
}

type mlComputeBackend struct{}

var defaultComputeBackend ComputeBackend = mlComputeBackend{}

// NewMLComputeBackend returns the default compute backend backed by internal/ml.
// Custom backends can wrap this and override only selected methods.
func NewMLComputeBackend() ComputeBackend {
	return mlComputeBackend{}
}

func (mlComputeBackend) Conv1D(ctx context.Context, x, w, b ml.Tensor, stride int) (ml.Tensor, error) {
	return ml.Conv1D(ctx, x, w, b, stride)
}

func (mlComputeBackend) GELU(x ml.Tensor) ml.Tensor {
	return ml.GELU(x)
}

func (mlComputeBackend) GELUInPlace(x ml.Tensor) {
	ml.GELUInPlace(x)
}

func (mlComputeBackend) Add(a, b ml.Tensor) ml.Tensor {
	return ml.Add(a, b)
}

func (mlComputeBackend) LayerNorm(x, weight, bias ml.Tensor, eps float32) ml.Tensor {
	return ml.LayerNorm(x, weight, bias, eps)
}

func (mlComputeBackend) LayerNormInto(dst, x, weight, bias ml.Tensor, eps float32) error {
	return ml.LayerNormInto(dst, x, weight, bias, eps)
}

func (mlComputeBackend) Transpose(x ml.Tensor, axes ...int) ml.Tensor {
	return ml.Transpose(x, axes...)
}

func (mlComputeBackend) ScaledDotProductAttention(ctx context.Context, q, k, v ml.Tensor, causal, returnScores bool) (ml.Tensor, ml.Tensor, error) {
	return ml.ScaledDotProductAttention(ctx, q, k, v, causal, returnScores)
}

func (mlComputeBackend) ScaledDotProductAttentionInto(ctx context.Context, q, k, v ml.Tensor, causal bool, out, scratch, scores ml.Tensor) error {
	return ml.ScaledDotProductAttentionInto(ctx, q, k, v, causal, out, scratch, scores)
}

func (mlComputeBackend) MatMulTransB(ctx context.Context, a, b ml.Tensor) (ml.Tensor, error) {
	return ml.MatMulTransB(ctx, a, b)
}

func (mlComputeBackend) MatMulTransBInto(ctx context.Context, a, b, out ml.Tensor) error {
	return ml.MatMulTransBInto(ctx, a, b, out)
}

func (mlComputeBackend) MatMulQuantTransBInto(ctx context.Context, a ml.Tensor, b ml.QuantizedMatrix, out ml.Tensor) error {
	return ml.MatMulQuantTransBInto(ctx, a, b, out)
}

func (mlComputeBackend) ShouldUseQuantMatMul(m, k, n int, qtype uint32) bool {
	return ml.ShouldUseQuantMatMul(m, k, n, qtype)
}

func (e *WhisperEncoder) computeBackend() ComputeBackend {
	if e != nil && e.backend != nil {
		return e.backend
	}
	return defaultComputeBackend
}

func (d *WhisperDecoder) computeBackend() ComputeBackend {
	if d != nil && d.backend != nil {
		return d.backend
	}
	return defaultComputeBackend
}
