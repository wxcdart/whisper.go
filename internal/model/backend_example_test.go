package model_test

import (
	"context"
	"fmt"

	"github.com/whispergo/whisper.go/internal/ml"
	"github.com/whispergo/whisper.go/internal/model"
)

// countingBackend is a minimal custom backend template that overrides one method
// and delegates all other methods to the default ML backend.
type countingBackend struct {
	model.ComputeBackend
	matMulCalls int
}

func (b *countingBackend) MatMulTransBInto(ctx context.Context, a, w, out ml.Tensor) error {
	b.matMulCalls++
	return b.ComputeBackend.MatMulTransBInto(ctx, a, w, out)
}

func ExampleNewMLComputeBackend_wrapper() {
	base := model.NewMLComputeBackend()
	custom := &countingBackend{ComputeBackend: base}

	a := ml.New(1, 2)
	a.Data[0], a.Data[1] = 1, 2
	w := ml.New(1, 2)
	w.Data[0], w.Data[1] = 3, 4
	out := ml.New(1, 1)

	_ = custom.MatMulTransBInto(context.Background(), a, w, out)
	fmt.Println(custom.matMulCalls)

	// Output: 1
}
