package model

import (
	"context"

	"github.com/whispergo/whisper.go/internal/ml"
)

func matMulTransBMaybeQuant(ctx context.Context, backend ComputeBackend, a, w ml.Tensor, wq *ml.QuantizedMatrix) (ml.Tensor, error) {
	if wq != nil && (len(w.Shape) == 0 || backend.ShouldUseQuantMatMul(a.Shape[0], a.Shape[1], wq.Rows, wq.QuantType)) {
		out := ml.New(a.Shape[0], wq.Rows)
		if err := backend.MatMulQuantTransBInto(ctx, a, *wq, out); err != nil {
			return ml.Tensor{}, err
		}
		return out, nil
	}
	return backend.MatMulTransB(ctx, a, w)
}

func matMulTransBMaybeQuantInto(ctx context.Context, backend ComputeBackend, a, w ml.Tensor, wq *ml.QuantizedMatrix, out ml.Tensor) error {
	if wq != nil && (len(w.Shape) == 0 || backend.ShouldUseQuantMatMul(a.Shape[0], a.Shape[1], wq.Rows, wq.QuantType)) {
		return backend.MatMulQuantTransBInto(ctx, a, *wq, out)
	}
	return backend.MatMulTransBInto(ctx, a, w, out)
}
