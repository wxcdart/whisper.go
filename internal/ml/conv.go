package ml

import (
	"context"
	"fmt"
	"runtime"

	"golang.org/x/sync/errgroup"
)

// Conv1D applies a 1-D convolution with same-ish padding (padding = kernel/2).
//
// Shapes:
//
//	input:  [inC, T]
//	weight: [outC, inC, kernel]
//	bias:   [outC]
//	output: [outC, T'] where T' = (T + 2*(kernel/2) - kernel) / stride + 1
func Conv1D(ctx context.Context, input, weight, bias Tensor, stride int) (Tensor, error) {
	if len(input.Shape) != 2 {
		return Tensor{}, fmt.Errorf("ml: conv1d: input must be 2D [inC, T], got %v", input.Shape)
	}
	if len(weight.Shape) != 3 {
		return Tensor{}, fmt.Errorf("ml: conv1d: weight must be 3D [outC, inC, kernel], got %v", weight.Shape)
	}
	inC, T := input.Shape[0], input.Shape[1]
	outC, wInC, kernel := weight.Shape[0], weight.Shape[1], weight.Shape[2]
	if wInC != inC {
		return Tensor{}, fmt.Errorf("%w: conv1d: input inC=%d != weight inC=%d", ErrShapeMismatch, inC, wInC)
	}
	if len(bias.Shape) != 1 || bias.Shape[0] != outC {
		return Tensor{}, fmt.Errorf("%w: conv1d: bias shape %v != [%d]", ErrShapeMismatch, bias.Shape, outC)
	}
	pad := kernel / 2
	Tprime := (T+2*pad-kernel)/stride + 1
	out := New(outC, Tprime)

	numCPU := runtime.NumCPU()
	g, gctx := errgroup.WithContext(ctx)
	chunkSize := (outC + numCPU - 1) / numCPU
	for start := 0; start < outC; start += chunkSize {
		start, end := start, start+chunkSize
		if end > outC {
			end = outC
		}
		g.Go(func() error {
			for oc := start; oc < end; oc++ {
				if err := gctx.Err(); err != nil {
					return err
				}
				for t := 0; t < Tprime; t++ {
					sum := bias.Data[oc]
					for ic := 0; ic < inC; ic++ {
						for k := 0; k < kernel; k++ {
							inT := t*stride - pad + k
							if inT < 0 || inT >= T {
								continue // zero padding
							}
							sum += input.Data[ic*T+inT] * weight.Data[oc*inC*kernel+ic*kernel+k]
						}
					}
					out.Data[oc*Tprime+t] = sum
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return Tensor{}, fmt.Errorf("ml: conv1d: %w", err)
	}
	return out, nil
}
