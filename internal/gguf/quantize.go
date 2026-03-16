package gguf

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sync"

	"golang.org/x/sync/errgroup"
)

// QuantType identifies a quantisation format.
type QuantType uint32

const (
	QuantF32  QuantType = 0
	QuantF16  QuantType = 1
	QuantQ4_0 QuantType = 2
	QuantQ4_1 QuantType = 3
	QuantQ5_0 QuantType = 6
	QuantQ5_1 QuantType = 7
	QuantQ8_0 QuantType = 8
)

// String returns a human-readable name for the quantisation type.
func (q QuantType) String() string {
	switch q {
	case QuantF32:
		return "F32"
	case QuantF16:
		return "F16"
	case QuantQ4_0:
		return "Q4_0"
	case QuantQ4_1:
		return "Q4_1"
	case QuantQ5_0:
		return "Q5_0"
	case QuantQ5_1:
		return "Q5_1"
	case QuantQ8_0:
		return "Q8_0"
	default:
		return fmt.Sprintf("QUANT(%d)", uint32(q))
	}
}

// QuantizeFile reads src GGUF, re-quantises weight tensors to targetType, writes to dst.
// Non-weight tensors (biases, norms, embeddings) are kept in F32.
func QuantizeFile(ctx context.Context, src, dst string, targetType QuantType) error {
	f, err := Open(ctx, src)
	if err != nil {
		return fmt.Errorf("gguf: quantize: open %s: %w", src, err)
	}
	defer f.Close() //nolint:errcheck

	names := f.TensorNames()

	out := NewWritableFile()
	out.copyMetaFrom(f)

	type result struct {
		idx int
		rec *tensorRecord
	}

	results := make([]*tensorRecord, len(names))
	var mu sync.Mutex
	_ = mu // results indexed by i; no lock needed (each goroutine writes its own index)

	g, gctx := errgroup.WithContext(ctx)
	for i, name := range names {
		i, name := i, name
		g.Go(func() error {
			if err := gctx.Err(); err != nil {
				return err
			}
			data, shape, err := f.Tensor(gctx, name)
			if err != nil {
				return fmt.Errorf("gguf: quantize: %s: %w", name, err)
			}

			var (
				dtype QuantType
				qdata []byte
			)
			if shouldQuantize(name) {
				dtype = targetType
				qdata, err = applyQuant(data, targetType)
				if err != nil {
					return fmt.Errorf("gguf: quantize: %s: %w", name, err)
				}
			} else {
				dtype = QuantF32
				qdata = float32sToBytes(data)
			}

			shape64 := make([]uint64, len(shape))
			for j, s := range shape {
				shape64[j] = uint64(s)
			}
			results[i] = &tensorRecord{
				name:  name,
				shape: shape64,
				dtype: dtype,
				data:  qdata,
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	out.tensors = results
	return WriteFile(ctx, dst, out)
}

// applyQuant encodes data into the requested quantisation format.
func applyQuant(data []float32, t QuantType) ([]byte, error) {
	switch t {
	case QuantF32:
		return float32sToBytes(data), nil
	case QuantF16:
		return QuantizeF16(data), nil
	case QuantQ4_0:
		return QuantizeQ4_0(data), nil
	case QuantQ4_1:
		return QuantizeQ4_1(data), nil
	case QuantQ5_0:
		return QuantizeQ5_0(data), nil
	case QuantQ5_1:
		return QuantizeQ5_1(data), nil
	case QuantQ8_0:
		return QuantizeQ8_0(data), nil
	default:
		return nil, fmt.Errorf("unsupported quant type %d", t)
	}
}

// float32sToBytes serialises float32 slice as little-endian IEEE-754.
func float32sToBytes(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
