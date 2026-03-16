package ml

import (
	"context"
	"fmt"
	"math"
	"runtime"

	"golang.org/x/sync/errgroup"
)

// ScaledDotProductAttention computes Attention(Q, K, V) with optional causal masking.
//
// Shapes:
//
//	q: [heads, T_q, head_dim]
//	k: [heads, T_k, head_dim]
//	v: [heads, T_k, head_dim]
//
// Returns:
//
//	out:     [heads, T_q, head_dim]
//	weights: [heads, T_q, T_k] if returnWeights is true, else zero Tensor
func ScaledDotProductAttention(ctx context.Context, q, k, v Tensor, causal, returnWeights bool) (out Tensor, weights Tensor, err error) {
	if len(q.Shape) != 3 || len(k.Shape) != 3 || len(v.Shape) != 3 {
		return Tensor{}, Tensor{}, fmt.Errorf("%w: attention: q/k/v must be 3D", ErrShapeMismatch)
	}
	heads, Tq, headDim := q.Shape[0], q.Shape[1], q.Shape[2]
	if k.Shape[0] != heads || v.Shape[0] != heads {
		return Tensor{}, Tensor{}, fmt.Errorf("%w: attention: head count mismatch", ErrShapeMismatch)
	}
	Tk := k.Shape[1]
	if k.Shape[2] != headDim || v.Shape[1] != Tk || v.Shape[2] != headDim {
		return Tensor{}, Tensor{}, fmt.Errorf("%w: attention: k/v shape mismatch", ErrShapeMismatch)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	outTensor := New(heads, Tq, headDim)
	var wTensor Tensor
	if returnWeights {
		wTensor = New(heads, Tq, Tk)
	}

	numCPU := runtime.NumCPU()
	g, gctx := errgroup.WithContext(ctx)
	chunkSize := (heads + numCPU - 1) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	negInf := float32(math.Inf(-1))

	for start := 0; start < heads; start += chunkSize {
		start, end := start, start+chunkSize
		if end > heads {
			end = heads
		}
		g.Go(func() error {
			scores := make([]float32, Tq*Tk)
			for h := start; h < end; h++ {
				if err := gctx.Err(); err != nil {
					return err
				}
				qOff := h * Tq * headDim
				kOff := h * Tk * headDim
				vOff := h * Tk * headDim

				// scores[i,j] = dot(q[i], k[j]) * scale
				for i := 0; i < Tq; i++ {
					for j := 0; j < Tk; j++ {
						var dot float32
						for d := 0; d < headDim; d++ {
							dot += q.Data[qOff+i*headDim+d] * k.Data[kOff+j*headDim+d]
						}
						scores[i*Tk+j] = dot * scale
					}
				}

				// causal mask: positions j > i get -inf
				if causal {
					for i := 0; i < Tq; i++ {
						for j := i + 1; j < Tk; j++ {
							scores[i*Tk+j] = negInf
						}
					}
				}

				// softmax over last axis (Tk)
				for i := 0; i < Tq; i++ {
					row := scores[i*Tk : (i+1)*Tk]
					maxVal := row[0]
					for _, val := range row[1:] {
						if val > maxVal {
							maxVal = val
						}
					}
					var sum float32
					for j := range row {
						row[j] = float32(math.Exp(float64(row[j] - maxVal)))
						sum += row[j]
					}
					for j := range row {
						row[j] /= sum
					}
				}

				if returnWeights {
					wOff := h * Tq * Tk
					copy(wTensor.Data[wOff:wOff+Tq*Tk], scores)
				}

				// out[h,i,d] = sum_j scores[i,j] * v[h,j,d]
				oOff := h * Tq * headDim
				for i := 0; i < Tq; i++ {
					for d := 0; d < headDim; d++ {
						var sum float32
						for j := 0; j < Tk; j++ {
							sum += scores[i*Tk+j] * v.Data[vOff+j*headDim+d]
						}
						outTensor.Data[oOff+i*headDim+d] = sum
					}
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return Tensor{}, Tensor{}, fmt.Errorf("ml: attention: %w", err)
	}
	return outTensor, wTensor, nil
}
