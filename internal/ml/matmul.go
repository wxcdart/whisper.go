package ml

import (
	"context"
	"fmt"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// MatMul computes C = A @ B.
// Supports 2-D [M,K] x [K,N] → [M,N] and batched [B,M,K] x [B,K,N] → [B,M,N].
func MatMul(ctx context.Context, a, b Tensor) (Tensor, error) {
	switch {
	case len(a.Shape) == 2 && len(b.Shape) == 2:
		return matmul2D(ctx, a, b, false)
	case len(a.Shape) == 3 && len(b.Shape) == 3:
		return matmulBatched(ctx, a, b, false)
	default:
		return Tensor{}, fmt.Errorf("%w: MatMul: a=%v b=%v", ErrShapeMismatch, a.Shape, b.Shape)
	}
}

// MatMulTransB computes C = A @ Bᵀ.
// B is provided in transposed layout: [N,K] (2-D) or [B,N,K] (batched).
func MatMulTransB(ctx context.Context, a, b Tensor) (Tensor, error) {
	switch {
	case len(a.Shape) == 2 && len(b.Shape) == 2:
		return matmul2D(ctx, a, b, true)
	case len(a.Shape) == 3 && len(b.Shape) == 3:
		return matmulBatched(ctx, a, b, true)
	default:
		return Tensor{}, fmt.Errorf("%w: MatMulTransB: a=%v b=%v", ErrShapeMismatch, a.Shape, b.Shape)
	}
}

// matmul2D handles the 2-D case, optionally treating B as transposed.
func matmul2D(ctx context.Context, a, b Tensor, transB bool) (Tensor, error) {
	if err := ctx.Err(); err != nil {
		return Tensor{}, err
	}
	M, K := a.Shape[0], a.Shape[1]
	var N int
	tB := blas.NoTrans
	var bGeneral blas32.General
	if transB {
		// b is [N, K], we want to treat it as [K, N] transposed.
		// BLAS Gemm with blas.Trans expects the General struct to describe
		// the NON-transposed matrix.
		N = b.Shape[0]
		if b.Shape[1] != K {
			return Tensor{}, fmt.Errorf("%w: matmul2D transB: a K=%d != b cols=%d", ErrShapeMismatch, K, b.Shape[1])
		}
		tB = blas.Trans
		bGeneral = blas32.General{Rows: N, Cols: K, Data: b.Data, Stride: K}
	} else {
		// b is [K, N]
		if b.Shape[0] != K {
			return Tensor{}, fmt.Errorf("%w: matmul2D: a K=%d != b rows=%d", ErrShapeMismatch, K, b.Shape[0])
		}
		N = b.Shape[1]
		bGeneral = blas32.General{Rows: K, Cols: N, Data: b.Data, Stride: N}
	}
	c := New(M, N)

	blas32.Gemm(blas.NoTrans, tB, 1,
		blas32.General{Rows: M, Cols: K, Data: a.Data, Stride: K},
		bGeneral,
		0,
		blas32.General{Rows: M, Cols: N, Data: c.Data, Stride: N},
	)

	return c, nil
}

// matmulBatched handles the 3-D batched case, optionally treating B as transposed.
func matmulBatched(ctx context.Context, a, b Tensor, transB bool) (Tensor, error) {
	if a.Shape[0] != b.Shape[0] {
		return Tensor{}, fmt.Errorf("%w: matmulBatched: batch mismatch a=%d b=%d", ErrShapeMismatch, a.Shape[0], b.Shape[0])
	}
	B, M, K := a.Shape[0], a.Shape[1], a.Shape[2]
	var N int
	tB := blas.NoTrans
	if transB {
		// b is [B, N, K]
		if b.Shape[2] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched transB: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[1]
		tB = blas.Trans
	} else {
		// b is [B, K, N]
		if b.Shape[1] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[2]
	}
	c := New(B, M, N)

	for bIdx := 0; bIdx < B; bIdx++ {
		if err := ctx.Err(); err != nil {
			return Tensor{}, err
		}
		aOff := bIdx * M * K
		cOff := bIdx * M * N

		var bGeneral blas32.General
		if transB {
			bOff := bIdx * N * K
			bGeneral = blas32.General{Rows: N, Cols: K, Data: b.Data[bOff : bOff+N*K], Stride: K}
		} else {
			bOff := bIdx * K * N
			bGeneral = blas32.General{Rows: K, Cols: N, Data: b.Data[bOff : bOff+K*N], Stride: N}
		}

		blas32.Gemm(blas.NoTrans, tB, 1,
			blas32.General{Rows: M, Cols: K, Data: a.Data[aOff : aOff+M*K], Stride: K},
			bGeneral,
			0,
			blas32.General{Rows: M, Cols: N, Data: c.Data[cOff : cOff+M*N], Stride: N},
		)
	}

	return c, nil
}

