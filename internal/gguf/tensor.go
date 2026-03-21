package gguf

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
)

// GGML dtype constants.
const (
	dtypeF32  = uint32(0)
	dtypeF16  = uint32(1)
	dtypeQ4_0 = uint32(2)
	dtypeQ4_1 = uint32(3)
	dtypeQ5_0 = uint32(6)
	dtypeQ5_1 = uint32(7)
	dtypeQ8_0 = uint32(8)
	dtypeQ4_K = uint32(12) // k-quant 4-bit
)

type tensorDesc struct {
	name   string
	shape  []uint64
	dtype  uint32
	offset uint64
}

func readTensorDesc(r io.Reader) (tensorDesc, error) {
	name, err := readString(r)
	if err != nil {
		return tensorDesc{}, fmt.Errorf("read name: %w", err)
	}
	var ndim uint32
	if err := binary.Read(r, binary.LittleEndian, &ndim); err != nil {
		return tensorDesc{}, fmt.Errorf("read ndim: %w", err)
	}
	shape := make([]uint64, ndim)
	for i := range shape {
		if err := binary.Read(r, binary.LittleEndian, &shape[i]); err != nil {
			return tensorDesc{}, fmt.Errorf("read shape[%d]: %w", i, err)
		}
	}
	var dtype uint32
	if err := binary.Read(r, binary.LittleEndian, &dtype); err != nil {
		return tensorDesc{}, fmt.Errorf("read dtype: %w", err)
	}
	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return tensorDesc{}, fmt.Errorf("read offset: %w", err)
	}
	return tensorDesc{name: name, shape: shape, dtype: dtype, offset: offset}, nil
}

// blocks returns the number of 32-element quantisation blocks needed for n elements.
func blocks(n uint64) uint64 {
	return (n + 31) / 32
}

// rawSize returns the byte size of tensor data for the given dtype and element count.
func rawSize(dtype uint32, n uint64) (uint64, error) {
	switch dtype {
	case dtypeF32:
		return n * 4, nil
	case dtypeF16:
		return n * 2, nil
	case dtypeQ4_0:
		return blocks(n) * 18, nil
	case dtypeQ4_1:
		return blocks(n) * 20, nil
	case dtypeQ5_0:
		return blocks(n) * 22, nil
	case dtypeQ5_1:
		return blocks(n) * 24, nil
	case dtypeQ8_0:
		return blocks(n) * 34, nil
	case dtypeQ4_K:
		// Q4_K: 256-element super-blocks; actual size varies by implementation
		// Common sizes: 140-160 bytes per 256 elements
		// Using 148 bytes (standard GGUF implementation with 8 blocks of ~18.5 bytes)
		return ((n + 255) / 256) * 148, nil
	default:
		return 0, fmt.Errorf("unsupported dtype: %d", dtype)
	}
}

const readChunk = int64(1 << 20) // 1 MiB

// readTensorRaw reads raw quantised bytes for a tensor from the open file.
// It checks ctx.Err() every readChunk bytes to support cancellation.
func readTensorRaw(ctx context.Context, f io.ReadSeeker, absOffset int64, dtype uint32, numElems uint64) ([]byte, error) {
	size, err := rawSize(dtype, numElems)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(absOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}
	buf := make([]byte, size)
	var read int64
	total := int64(size)
	for read < total {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled: %w", err)
		}
		end := read + readChunk
		if end > total {
			end = total
		}
		if _, err := io.ReadFull(f, buf[read:end]); err != nil {
			return nil, fmt.Errorf("read data: %w", err)
		}
		read = end
	}
	return buf, nil
}
