package ml

import (
	"context"
	"math"
	"testing"
)

func quantizeQ8Row(vals []float32) []byte {
	n := len(vals)
	nBlocks := (n + q80BlockElems - 1) / q80BlockElems
	out := make([]byte, nBlocks*q80BlockBytes)
	for b := 0; b < nBlocks; b++ {
		start := b * q80BlockElems
		end := start + q80BlockElems
		if end > n {
			end = n
		}
		absmax := float32(0)
		for i := start; i < end; i++ {
			a := float32(math.Abs(float64(vals[i])))
			if a > absmax {
				absmax = a
			}
		}
		scale := float32(0)
		if absmax > 0 {
			scale = absmax / 127.0
		}
		h := float32ToF16Local(scale)
		bo := b * q80BlockBytes
		out[bo] = byte(h)
		out[bo+1] = byte(h >> 8)
		for i := 0; i < q80BlockElems; i++ {
			idx := start + i
			q := int8(0)
			if idx < n && scale != 0 {
				v := int32(math.Round(float64(vals[idx] / scale)))
				if v < -128 {
					v = -128
				}
				if v > 127 {
					v = 127
				}
				q = int8(v)
			}
			out[bo+2+i] = byte(q)
		}
	}
	return out
}

func float32ToF16Local(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 31)
	exp := int(bits>>23) & 0xFF
	mant := bits & 0x7FFFFF
	switch {
	case exp == 255:
		if mant != 0 {
			return sign<<15 | 0x7E00
		}
		return sign<<15 | 0x7C00
	case exp == 0:
		return sign << 15
	}
	exp16 := exp - 127 + 15
	if exp16 >= 31 {
		return sign<<15 | 0x7C00
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign << 15
		}
		shift := uint(1 - exp16)
		mant16 := uint16((mant | 0x800000) >> (13 + shift))
		return sign<<15 | mant16
	}
	mant16 := uint16(mant >> 13)
	if (mant>>12)&1 == 1 {
		mant16++
	}
	if mant16 >= 0x400 {
		mant16 = 0
		exp16++
		if exp16 >= 31 {
			return sign<<15 | 0x7C00
		}
	}
	return sign<<15 | uint16(exp16)<<10 | mant16
}

func TestDotQ8_0F32(t *testing.T) {
	f := make([]float32, 64)
	g := make([]float32, 64)
	for i := range f {
		f[i] = float32(i%7) - 3
		g[i] = float32((i*3)%11) - 5
	}
	q := quantizeQ8Row(g)
	dotQ, err := DotQ8_0F32(q, f)
	if err != nil {
		t.Fatal(err)
	}
	var dotRef float32
	for i := range f {
		dotRef += f[i] * g[i]
	}
	if !approxEq(dotQ, dotRef, 5e-1) {
		t.Fatalf("dot mismatch: q8=%v ref=%v", dotQ, dotRef)
	}
}

func TestMatMulQ8_0TransBInto(t *testing.T) {
	ctx := context.Background()
	m, k, n := 2, 64, 3
	a := New(m, k)
	for i := range a.Data {
		a.Data[i] = float32((i%13)-6) * 0.1
	}
	bF := New(n, k)
	for i := range bF.Data {
		bF.Data[i] = float32((i%9)-4) * 0.2
	}

	blocksPerRow := (k + q80BlockElems - 1) / q80BlockElems
	rowBytes := blocksPerRow * q80BlockBytes
	bQ := make([]byte, n*rowBytes)
	for r := 0; r < n; r++ {
		row := bF.Data[r*k : (r+1)*k]
		copy(bQ[r*rowBytes:(r+1)*rowBytes], quantizeQ8Row(row))
	}

	got := New(m, n)
	if err := MatMulQ8_0TransBInto(ctx, a, bQ, n, k, got); err != nil {
		t.Fatal(err)
	}
	ref, err := MatMulTransB(ctx, a, bF)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ref.Data {
		if !approxEq(got.Data[i], ref.Data[i], 1.0) {
			t.Fatalf("[%d] mismatch: got=%v ref=%v", i, got.Data[i], ref.Data[i])
		}
	}
}
