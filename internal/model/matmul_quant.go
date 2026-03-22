package model

import (
	"context"

	"github.com/whispergo/whisper.go/internal/ml"
)

func matMulTransBMaybeQuant(ctx context.Context, a, w ml.Tensor, wq *ml.QuantizedMatrix) (ml.Tensor, error) {
	if wq != nil {
		out := ml.New(a.Shape[0], wq.Rows)
		if err := ml.MatMulQuantTransBInto(ctx, a, *wq, out); err != nil {
			return ml.Tensor{}, err
		}
		return out, nil
	}
	return ml.MatMulTransB(ctx, a, w)
}

func matMulTransBMaybeQuantInto(ctx context.Context, a, w ml.Tensor, wq *ml.QuantizedMatrix, out ml.Tensor) error {
	if wq != nil {
		return ml.MatMulQuantTransBInto(ctx, a, *wq, out)
	}
	return ml.MatMulTransBInto(ctx, a, w, out)
}
