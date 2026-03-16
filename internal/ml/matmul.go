package ml

import (
	"context"
	"fmt"
	"runtime"

	"golang.org/x/sync/errgroup"
)

// MatMul computes C = A @ B.
// Supports 2-D [M,K] x [K,N] → [M,N] and batched [B,M,K] x [B,K,N] → [B,M,N].
// Parallelised over rows using errgroup.
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
	M, K := a.Shape[0], a.Shape[1]
	var N int
	if transB {
		// b is [N, K]
		if b.Shape[1] != K {
			return Tensor{}, fmt.Errorf("%w: matmul2D transB: a K=%d != b cols=%d", ErrShapeMismatch, K, b.Shape[1])
		}
		N = b.Shape[0]
	} else {
		// b is [K, N]
		if b.Shape[0] != K {
			return Tensor{}, fmt.Errorf("%w: matmul2D: a K=%d != b rows=%d", ErrShapeMismatch, K, b.Shape[0])
		}
		N = b.Shape[1]
	}
	c := New(M, N)
	numCPU := runtime.NumCPU()
	g, gctx := errgroup.WithContext(ctx)
	chunkSize := (M + numCPU - 1) / numCPU
	for start := 0; start < M; start += chunkSize {
		start, end := start, start+chunkSize
		if end > M {
			end = M
		}
		g.Go(func() error {
			for i := start; i < end; i++ {
				if err := gctx.Err(); err != nil {
					return err
				}
				for j := 0; j < N; j++ {
					var sum float32
					if transB {
						for k := 0; k < K; k++ {
							sum += a.Data[i*K+k] * b.Data[j*K+k]
						}
					} else {
						for k := 0; k < K; k++ {
							sum += a.Data[i*K+k] * b.Data[k*N+j]
						}
					}
					c.Data[i*N+j] = sum
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return Tensor{}, fmt.Errorf("ml: matmul: %w", err)
	}
	return c, nil
}

// matmulBatched handles the 3-D batched case, optionally treating B as transposed.
func matmulBatched(ctx context.Context, a, b Tensor, transB bool) (Tensor, error) {
	if a.Shape[0] != b.Shape[0] {
		return Tensor{}, fmt.Errorf("%w: matmulBatched: batch mismatch a=%d b=%d", ErrShapeMismatch, a.Shape[0], b.Shape[0])
	}
	B, M, K := a.Shape[0], a.Shape[1], a.Shape[2]
	var N int
	if transB {
		// b is [B, N, K]
		if b.Shape[2] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched transB: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[1]
	} else {
		// b is [B, K, N]
		if b.Shape[1] != K {
			return Tensor{}, fmt.Errorf("%w: matmulBatched: K mismatch", ErrShapeMismatch)
		}
		N = b.Shape[2]
	}
	c := New(B, M, N)
	numCPU := runtime.NumCPU()
	totalRows := B * M
	g, gctx := errgroup.WithContext(ctx)
	chunkSize := (totalRows + numCPU - 1) / numCPU
	for start := 0; start < totalRows; start += chunkSize {
		start, end := start, start+chunkSize
		if end > totalRows {
			end = totalRows
		}
		g.Go(func() error {
			for row := start; row < end; row++ {
				if err := gctx.Err(); err != nil {
					return err
				}
				batch := row / M
				i := row % M
				aOff := batch*M*K + i*K
				cOff := batch*M*N + i*N
				for j := 0; j < N; j++ {
					var sum float32
					if transB {
						bOff := batch*N*K + j*K
						for k := 0; k < K; k++ {
							sum += a.Data[aOff+k] * b.Data[bOff+k]
						}
					} else {
						bOff := batch * K * N
						for k := 0; k < K; k++ {
							sum += a.Data[aOff+k] * b.Data[bOff+k*N+j]
						}
					}
					c.Data[cOff+j] = sum
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return Tensor{}, fmt.Errorf("ml: matmul_batched: %w", err)
	}
	return c, nil
}
