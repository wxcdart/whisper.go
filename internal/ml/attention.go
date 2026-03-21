package ml

import (
	"context"
	"fmt"
	"math"
	"runtime"

	"golang.org/x/sync/errgroup"
)

// attentionUseFastExp controls whether attention softmax uses the fast exp path.
// It defaults to true for performance and can be toggled in benchmarks/tests.
var attentionUseFastExp = true

// SetFastSoftmaxEnabled toggles the fast-exp softmax implementation used by attention.
// It returns the previous value.
func SetFastSoftmaxEnabled(enabled bool) (previous bool) {
	previous = attentionUseFastExp
	attentionUseFastExp = enabled
	return previous
}

const (
	invLn2 float32 = 1.4426950408889634 // 1/ln(2)
	ln2    float32 = 0.6931471805599453
)

var (
	softmaxRowFastKernel  = softmaxRowFastInPlaceGeneric
	softmaxRowExactKernel = softmaxRowExactInPlaceGeneric
)

func fastExpApprox(x float32) float32 {
	// Softmax uses x <= 0 (after subtracting max); large negative values are effectively zero.
	if x <= -20 {
		return 0
	}

	y := x * invLn2
	k := int(y)
	if float32(k) > y {
		k--
	}
	r := x - float32(k)*ln2

	// 4th-order polynomial for exp(r) on [0, ln2).
	r2 := r * r
	r3 := r2 * r
	r4 := r3 * r
	poly := 1 + r + 0.5*r2 + (1.0/6.0)*r3 + (1.0/24.0)*r4

	if k < -126 {
		return 0
	}
	if k > 127 {
		return math.MaxFloat32
	}
	twoPowK := math.Float32frombits(uint32(k+127) << 23)
	return poly * twoPowK
}

func rowMaxUnrolled(row []float32) float32 {
	maxVal := row[0]
	i := 1
	n := len(row)
	for ; i+3 < n; i += 4 {
		v0 := row[i]
		v1 := row[i+1]
		v2 := row[i+2]
		v3 := row[i+3]
		if v0 > maxVal {
			maxVal = v0
		}
		if v1 > maxVal {
			maxVal = v1
		}
		if v2 > maxVal {
			maxVal = v2
		}
		if v3 > maxVal {
			maxVal = v3
		}
	}
	for ; i < n; i++ {
		if row[i] > maxVal {
			maxVal = row[i]
		}
	}
	return maxVal
}

func softmaxRowFastInPlaceGeneric(row []float32) {
	maxVal := rowMaxUnrolled(row)
	n := len(row)
	var sum float32
	i := 0
	for ; i+3 < n; i += 4 {
		x0 := fastExpApprox(row[i] - maxVal)
		x1 := fastExpApprox(row[i+1] - maxVal)
		x2 := fastExpApprox(row[i+2] - maxVal)
		x3 := fastExpApprox(row[i+3] - maxVal)
		row[i] = x0
		row[i+1] = x1
		row[i+2] = x2
		row[i+3] = x3
		sum += x0 + x1 + x2 + x3
	}
	for ; i < n; i++ {
		x := fastExpApprox(row[i] - maxVal)
		row[i] = x
		sum += x
	}
	if sum == 0 {
		inv := 1.0 / float32(n)
		for i := range row {
			row[i] = inv
		}
		return
	}
	invSum := 1 / sum
	for i := 0; i+3 < n; i += 4 {
		row[i] *= invSum
		row[i+1] *= invSum
		row[i+2] *= invSum
		row[i+3] *= invSum
	}
	for i := n &^ 3; i < n; i++ {
		row[i] *= invSum
	}
}

func softmaxRowExactInPlaceGeneric(row []float32) {
	maxVal := rowMaxUnrolled(row)
	n := len(row)
	var sum float32
	i := 0
	for ; i+3 < n; i += 4 {
		x0 := float32(math.Exp(float64(row[i] - maxVal)))
		x1 := float32(math.Exp(float64(row[i+1] - maxVal)))
		x2 := float32(math.Exp(float64(row[i+2] - maxVal)))
		x3 := float32(math.Exp(float64(row[i+3] - maxVal)))
		row[i] = x0
		row[i+1] = x1
		row[i+2] = x2
		row[i+3] = x3
		sum += x0 + x1 + x2 + x3
	}
	for ; i < n; i++ {
		x := float32(math.Exp(float64(row[i] - maxVal)))
		row[i] = x
		sum += x
	}
	if sum == 0 {
		inv := 1.0 / float32(n)
		for i := range row {
			row[i] = inv
		}
		return
	}
	invSum := 1 / sum
	for i := 0; i+3 < n; i += 4 {
		row[i] *= invSum
		row[i+1] *= invSum
		row[i+2] *= invSum
		row[i+3] *= invSum
	}
	for i := n &^ 3; i < n; i++ {
		row[i] *= invSum
	}
}

func softmaxRowFastInPlace(row []float32) {
	softmaxRowFastKernel(row)
}

func softmaxRowExactInPlace(row []float32) {
	softmaxRowExactKernel(row)
}

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
func ScaledDotProductAttentionInto(ctx context.Context, q, k, v Tensor, causal bool, out, weights, scoresScratch Tensor) error {
	if len(q.Shape) != 3 || len(k.Shape) != 3 || len(v.Shape) != 3 {
		return fmt.Errorf("%w: attention: q/k/v must be 3D", ErrShapeMismatch)
	}
	heads, Tq, headDim := q.Shape[0], q.Shape[1], q.Shape[2]
	if k.Shape[0] != heads || v.Shape[0] != heads {
		return fmt.Errorf("%w: attention: head count mismatch", ErrShapeMismatch)
	}
	Tk := k.Shape[1]
	if k.Shape[2] != headDim || v.Shape[1] != Tk || v.Shape[2] != headDim {
		return fmt.Errorf("%w: attention: k/v shape mismatch", ErrShapeMismatch)
	}

	if len(out.Shape) != 3 || out.Shape[0] != heads || out.Shape[1] != Tq || out.Shape[2] != headDim {
		return fmt.Errorf("%w: attention: out shape mismatch: got %v, want [%d %d %d]", ErrShapeMismatch, out.Shape, heads, Tq, headDim)
	}

	returnWeights := len(weights.Shape) != 0
	if returnWeights {
		if len(weights.Shape) != 3 || weights.Shape[0] != heads || weights.Shape[1] != Tq || weights.Shape[2] != Tk {
			return fmt.Errorf("%w: attention: weights shape mismatch: got %v, want [%d %d %d]", ErrShapeMismatch, weights.Shape, heads, Tq, Tk)
		}
	}

	if len(scoresScratch.Shape) != 3 || scoresScratch.Shape[0] != heads || scoresScratch.Shape[1] != Tq || scoresScratch.Shape[2] != Tk {
		return fmt.Errorf("%w: attention: scores scratch shape mismatch: got %v, want [%d %d %d]", ErrShapeMismatch, scoresScratch.Shape, heads, Tq, Tk)
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	negInf := float32(math.Inf(-1))

	processHeadRange := func(start, end int) error {
		for h := start; h < end; h++ {
			if err := ctx.Err(); err != nil {
				return err
			}

			qOff := h * Tq * headDim
			kOff := h * Tk * headDim
			vOff := h * Tk * headDim

			scoresOff := h * Tq * Tk
			scores := scoresScratch.Data[scoresOff : scoresOff+Tq*Tk]

			// Q: [Tq, headDim], K: [Tk, headDim] -> scores: [Tq, Tk]
			qTensor := Tensor{Shape: []int{Tq, headDim}, Data: q.Data[qOff : qOff+Tq*headDim]}
			kTensor := Tensor{Shape: []int{Tk, headDim}, Data: k.Data[kOff : kOff+Tk*headDim]}
			scoresTensor := Tensor{Shape: []int{Tq, Tk}, Data: scores}
			err := MatMulTransBInto(ctx, qTensor, kTensor, scoresTensor)
			if err != nil {
				return err
			}

			for i := range scores {
				scores[i] *= scale
			}

			if causal {
				for i := 0; i < Tq; i++ {
					for j := i + 1; j < Tk; j++ {
						scores[i*Tk+j] = negInf
					}
				}
			}

			for i := 0; i < Tq; i++ {
				row := scores[i*Tk : (i+1)*Tk]
				if attentionUseFastExp {
					softmaxRowFastInPlace(row)
				} else {
					softmaxRowExactInPlace(row)
				}
			}

			if returnWeights {
				copy(weights.Data[scoresOff:scoresOff+Tq*Tk], scores)
			}

			outScoresTensor := Tensor{Shape: []int{Tq, Tk}, Data: scores}
			vTensor := Tensor{Shape: []int{Tk, headDim}, Data: v.Data[vOff : vOff+Tk*headDim]}
			outOff := h * Tq * headDim
			headOut := Tensor{Shape: []int{Tq, headDim}, Data: out.Data[outOff : outOff+Tq*headDim]}
			err = MatMulInto(ctx, outScoresTensor, vTensor, headOut)
			if err != nil {
				return err
			}
		}
		return nil
	}

	// Token-wise decoder attention typically has Tq=1 and small head counts.
	// Avoid goroutine/errgroup overhead in this hot path.
	if Tq == 1 || heads <= 4 {
		if err := processHeadRange(0, heads); err != nil {
			return fmt.Errorf("ml: attention: %w", err)
		}
		return nil
	}

	numCPU := runtime.NumCPU()
	g, gctx := errgroup.WithContext(ctx)
	chunkSize := (heads + numCPU - 1) / numCPU
	if chunkSize < 1 {
		chunkSize = 1
	}

	for start := 0; start < heads; start += chunkSize {
		start, end := start, start+chunkSize
		if end > heads {
			end = heads
		}
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			return processHeadRange(start, end)
		})
	}

	if err := g.Wait(); err != nil {
		return fmt.Errorf("ml: attention: %w", err)
	}

	return nil
}

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

	out = New(heads, Tq, headDim)
	var w Tensor
	if returnWeights {
		w = New(heads, Tq, Tk)
	}
	scores := New(heads, Tq, Tk)

	if err := ScaledDotProductAttentionInto(ctx, q, k, v, causal, out, w, scores); err != nil {
		return Tensor{}, Tensor{}, err
	}

	return out, w, nil
}
