package ml

import (
	"context"
	"fmt"
)

// MatMul computes C = A @ B.
// Supports 2-D [M,K] x [K,N] → [M,N] and batched [B,M,K] x [B,K,N] → [B,M,N].
func MatMul(ctx context.Context, a, b Tensor) (Tensor, error) {
	switch {
	case len(a.Shape) == 2 && len(b.Shape) == 2:
		M, N := a.Shape[0], b.Shape[1]
		c := New(M, N)
		if err := matmul2DInto(ctx, a, b, false, c); err != nil {
			return Tensor{}, err
		}
		return c, nil
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
		M, N := a.Shape[0], b.Shape[0]
		c := New(M, N)
		if err := matmul2DInto(ctx, a, b, true, c); err != nil {
			return Tensor{}, err
		}
		return c, nil
	case len(a.Shape) == 3 && len(b.Shape) == 3:
		return matmulBatched(ctx, a, b, true)
	default:
		return Tensor{}, fmt.Errorf("%w: MatMulTransB: a=%v b=%v", ErrShapeMismatch, a.Shape, b.Shape)
	}
}

// MatMulInto computes out = A @ B for 2-D tensors without allocating output.
func MatMulInto(ctx context.Context, a, b, out Tensor) error {
	if len(a.Shape) != 2 || len(b.Shape) != 2 || len(out.Shape) != 2 {
		return fmt.Errorf("%w: MatMulInto: a=%v b=%v out=%v", ErrShapeMismatch, a.Shape, b.Shape, out.Shape)
	}
	return matmul2DInto(ctx, a, b, false, out)
}

// MatMulTransBInto computes out = A @ B^T for 2-D tensors without allocating output.
func MatMulTransBInto(ctx context.Context, a, b, out Tensor) error {
	if len(a.Shape) != 2 || len(b.Shape) != 2 || len(out.Shape) != 2 {
		return fmt.Errorf("%w: MatMulTransBInto: a=%v b=%v out=%v", ErrShapeMismatch, a.Shape, b.Shape, out.Shape)
	}
	return matmul2DInto(ctx, a, b, true, out)
}

// matmul2DInto handles the 2-D case, optionally treating B as transposed, writing into out.
func matmul2DInto(ctx context.Context, a, b Tensor, transB bool, out Tensor) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	M, K := a.Shape[0], a.Shape[1]
	var N int
	tB := cblasTranspose(transB)
	var ldb int32
	if transB {
		// b is [N, K], treated as [K, N] transposed.
		N = b.Shape[0]
		if b.Shape[1] != K {
			return fmt.Errorf("%w: matmul2D transB: a K=%d != b cols=%d", ErrShapeMismatch, K, b.Shape[1])
		}
		ldb = int32(K)
	} else {
		// b is [K, N]
		if b.Shape[0] != K {
			return fmt.Errorf("%w: matmul2D: a K=%d != b rows=%d", ErrShapeMismatch, K, b.Shape[0])
		}
		N = b.Shape[1]
		ldb = int32(N)
	}

	if out.Shape[0] != M || out.Shape[1] != N {
		return fmt.Errorf("%w: matmul2DInto out shape: got %v want [%d %d]", ErrShapeMismatch, out.Shape, M, N)
	}

	sgemmImpl(
		cblasNoTrans, tB,
		int32(M), int32(N), int32(K),
		1,
		a.Data, int32(K),
		b.Data, ldb,
		0,
		out.Data, int32(N),
	)

	return nil
}

// matmulBatched handles the 3-D batched case, optionally treating B as transposed.
func matmulBatched(ctx context.Context, a, b Tensor, transB bool) (Tensor, error) {
	if a.Shape[0] != b.Shape[0] {
		return Tensor{}, fmt.Errorf("%w: matmulBatched: batch mismatch a=%d b=%d", ErrShapeMismatch, a.Shape[0], b.Shape[0])
	}
	B, M, K := a.Shape[0], a.Shape[1], a.Shape[2]
	var N int
	tB := cblasTranspose(transB)
	var ldb int32
	if transB {
		// b is [B, N, K]
		if b.Shape[2] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched transB: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[1]
		ldb = int32(K)
	} else {
		// b is [B, K, N]
		if b.Shape[1] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[2]
		ldb = int32(N)
	}
	c := New(B, M, N)

	for bIdx := 0; bIdx < B; bIdx++ {
		if err := ctx.Err(); err != nil {
			return Tensor{}, err
		}
		aOff := bIdx * M * K
		cOff := bIdx * M * N

		var bSlice []float32
		if transB {
			bOff := bIdx * N * K
			bSlice = b.Data[bOff : bOff+N*K]
		} else {
			bOff := bIdx * K * N
			bSlice = b.Data[bOff : bOff+K*N]
		}

		sgemmImpl(
			cblasNoTrans, tB,
			int32(M), int32(N), int32(K),
			1,
			a.Data[aOff:aOff+M*K], int32(K),
			bSlice, ldb,
			0,
			c.Data[cOff:cOff+M*N], int32(N),
		)
	}

	return c, nil
}
