package ml

import (
	"context"
	"fmt"
	"math"
)

const (
	q80BlockElems = 32
	q80BlockBytes = 34
)

// DotQ8_0F32 computes dot(q8, f32) where q8 is GGUF Q8_0 block-encoded data.
// q8 data layout per 32 elements: f16 scale (little-endian) + 32 int8 values.
func DotQ8_0F32(q8 []byte, f []float32) (float32, error) {
	if len(f) == 0 {
		return 0, nil
	}
	n := len(f)
	nBlocks := (n + q80BlockElems - 1) / q80BlockElems
	if len(q8) < nBlocks*q80BlockBytes {
		return 0, fmt.Errorf("%w: DotQ8_0F32: q8 bytes short: got=%d need=%d", ErrShapeMismatch, len(q8), nBlocks*q80BlockBytes)
	}

	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q80BlockBytes
		scale := f16ToF32Local(uint16(q8[bo]) | uint16(q8[bo+1])<<8)
		v := q8[bo+2 : bo+2+q80BlockElems]
		start := b * q80BlockElems
		end := start + q80BlockElems
		if end > n {
			end = n
		}
		for i := start; i < end; i++ {
			sum += scale * float32(int8(v[i-start])) * f[i]
		}
	}
	return sum, nil
}

// MatMulQ8_0TransBInto computes out = A @ B^T where:
// - A is [M, K] float32 tensor
// - B is [N, K] encoded as row-wise Q8_0 bytes in bQ8
// - out is [M, N] float32 tensor
// This is a narrow quantized path for decoder-like projection matrices.
func MatMulQ8_0TransBInto(ctx context.Context, a Tensor, bQ8 []byte, nRowsB, k int, out Tensor) error {
	if len(a.Shape) != 2 || len(out.Shape) != 2 {
		return fmt.Errorf("%w: MatMulQ8_0TransBInto: a=%v out=%v", ErrShapeMismatch, a.Shape, out.Shape)
	}
	m := a.Shape[0]
	if a.Shape[1] != k {
		return fmt.Errorf("%w: MatMulQ8_0TransBInto: a.K=%d != k=%d", ErrShapeMismatch, a.Shape[1], k)
	}
	if out.Shape[0] != m || out.Shape[1] != nRowsB {
		return fmt.Errorf("%w: MatMulQ8_0TransBInto: out=%v want=[%d %d]", ErrShapeMismatch, out.Shape, m, nRowsB)
	}
	blocksPerRow := (k + q80BlockElems - 1) / q80BlockElems
	rowBytes := blocksPerRow * q80BlockBytes
	if len(bQ8) < nRowsB*rowBytes {
		return fmt.Errorf("%w: MatMulQ8_0TransBInto: bQ8 short: got=%d need=%d", ErrShapeMismatch, len(bQ8), nRowsB*rowBytes)
	}

	for i := 0; i < m; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		aRow := a.Data[i*k : (i+1)*k]
		for j := 0; j < nRowsB; j++ {
			bRow := bQ8[j*rowBytes : (j+1)*rowBytes]
			dot, err := DotQ8_0F32(bRow, aRow)
			if err != nil {
				return err
			}
			out.Data[i*nRowsB+j] = dot
		}
	}
	return nil
}

func f16ToF32Local(h uint16) float32 {
	s := uint32(h>>15) << 31
	e := uint32((h >> 10) & 0x1F)
	m := uint32(h & 0x3FF)
	switch e {
	case 0:
		if m == 0 {
			return float32FromBits(s)
		}
		k := uint32(1)
		m <<= 1
		for m&0x400 == 0 {
			m <<= 1
			k++
		}
		m &= 0x3FF
		return float32FromBits(s | ((113 - k) << 23) | (m << 13))
	case 31:
		return float32FromBits(s | 0x7F800000 | (m << 13))
	default:
		return float32FromBits(s | ((e + 112) << 23) | (m << 13))
	}
}

func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
