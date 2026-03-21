package ml

import (
	"context"
	"math/rand"
	"testing"
)

func TestScaledDotProductAttention_FastSoftmaxCloseToExact(t *testing.T) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(7))

	q := New(4, 32, 64)
	k := New(4, 32, 64)
	v := New(4, 32, 64)
	for i := range q.Data {
		q.Data[i] = (rng.Float32() - 0.5) * 2
	}
	for i := range k.Data {
		k.Data[i] = (rng.Float32() - 0.5) * 2
	}
	for i := range v.Data {
		v.Data[i] = (rng.Float32() - 0.5) * 2
	}

	prev := SetFastSoftmaxEnabled(false)
	exactOut, _, err := ScaledDotProductAttention(ctx, q, k, v, false, false)
	if err != nil {
		t.Fatalf("exact attention failed: %v", err)
	}

	SetFastSoftmaxEnabled(true)
	fastOut, _, err := ScaledDotProductAttention(ctx, q, k, v, false, false)
	if err != nil {
		t.Fatalf("fast attention failed: %v", err)
	}
	SetFastSoftmaxEnabled(prev)

	if len(exactOut.Data) != len(fastOut.Data) {
		t.Fatalf("shape mismatch: exact=%v fast=%v", exactOut.Shape, fastOut.Shape)
	}

	var maxAbs float32
	var mse float64
	for i := range exactOut.Data {
		d := exactOut.Data[i] - fastOut.Data[i]
		if d < 0 {
			d = -d
		}
		if d > maxAbs {
			maxAbs = d
		}
		mse += float64(d * d)
	}
	mse /= float64(len(exactOut.Data))

	if maxAbs > 0.02 {
		t.Fatalf("fast softmax deviation too high: maxAbs=%.6f mse=%.8f", maxAbs, mse)
	}
}
