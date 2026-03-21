package ml

import (
	"context"
	"math/rand"
	"testing"
)

func makeRandTensor(rng *rand.Rand, shape ...int) Tensor {
	t := New(shape...)
	for i := range t.Data {
		t.Data[i] = (rng.Float32() - 0.5) * 2
	}
	return t
}

func BenchmarkScaledDotProductAttention_SoftmaxModes(b *testing.B) {
	ctx := context.Background()
	rng := rand.New(rand.NewSource(1))

	const (
		heads   = 6
		tq      = 96
		tk      = 96
		headDim = 64
	)

	q := makeRandTensor(rng, heads, tq, headDim)
	k := makeRandTensor(rng, heads, tk, headDim)
	v := makeRandTensor(rng, heads, tk, headDim)

	b.Run("exact_exp", func(b *testing.B) {
		prev := SetFastSoftmaxEnabled(false)
		defer SetFastSoftmaxEnabled(prev)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, err := ScaledDotProductAttention(ctx, q, k, v, false, false)
			if err != nil {
				b.Fatalf("attention failed: %v", err)
			}
		}
	})

	b.Run("fast_exp", func(b *testing.B) {
		prev := SetFastSoftmaxEnabled(true)
		defer SetFastSoftmaxEnabled(prev)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _, err := ScaledDotProductAttention(ctx, q, k, v, false, false)
			if err != nil {
				b.Fatalf("attention failed: %v", err)
			}
		}
	})
}
