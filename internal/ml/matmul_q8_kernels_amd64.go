//go:build amd64

package ml

import (
	"fmt"
	"math/rand"
	"time"
	"unsafe"
)

//go:noescape
func dotInt8F32x32AVX2Asm(q *int8, f *float32) float32

func dotQ8_0F32AVX2(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q80BlockElems - 1) / q80BlockElems
	if len(q) < nBlocks*q80BlockBytes {
		return 0, fmt.Errorf("%w: DotQ8_0F32: q8 bytes short: got=%d need=%d", ErrShapeMismatch, len(q), nBlocks*q80BlockBytes)
	}

	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q80BlockBytes
		scale := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		start := b * q80BlockElems
		rem := n - start
		if rem >= q80BlockElems {
			qptr := (*int8)(unsafe.Pointer(&q[bo+2]))
			fptr := (*float32)(unsafe.Pointer(&f[start]))
			sum += scale * dotInt8F32x32AVX2Asm(qptr, fptr)
			continue
		}
		for i := 0; i < rem; i++ {
			sum += scale * float32(int8(q[bo+2+i])) * f[start+i]
		}
	}
	return sum, nil
}

func dotQ4_0F32AVX2(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q40BlockElems - 1) / q40BlockElems
	if len(q) < nBlocks*q40BlockBytes {
		return 0, fmt.Errorf("%w: dotQ4_0F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q40BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		start := b * 32
		rem := n - start
		if rem >= 32 {
			var tmp [32]int8
			for i := 0; i < 16; i++ {
				qb := q[bo+2+i]
				tmp[i*2] = int8((qb & 0x0F) - 8)
				tmp[i*2+1] = int8((qb >> 4) - 8)
			}
			fptr := (*float32)(unsafe.Pointer(&f[start]))
			sum += d * dotInt8F32x32AVX2Asm(&tmp[0], fptr)
			continue
		}
		for i := 0; i < rem; i++ {
			qb := q[bo+2+(i>>1)]
			var qv int32
			if i&1 == 0 {
				qv = int32(qb&0xF) - 8
			} else {
				qv = int32(qb>>4) - 8
			}
			sum += d * float32(qv) * f[start+i]
		}
	}
	return sum, nil
}

func dotQ4_1F32AVX2(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q41BlockElems - 1) / q41BlockElems
	if len(q) < nBlocks*q41BlockBytes {
		return 0, fmt.Errorf("%w: dotQ4_1F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q41BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		m := f16ToF32Local(uint16(q[bo+2]) | uint16(q[bo+3])<<8)
		start := b * 32
		rem := n - start
		if rem >= 32 {
			var tmp [32]int8
			for i := 0; i < 16; i++ {
				qb := q[bo+4+i]
				tmp[i*2] = int8(qb & 0x0F)
				tmp[i*2+1] = int8(qb >> 4)
			}
			fblk := f[start : start+32]
			fptr := (*float32)(unsafe.Pointer(&fblk[0]))
			s := dotInt8F32x32AVX2Asm(&tmp[0], fptr)
			var fsum float32
			for i := 0; i < 32; i++ {
				fsum += fblk[i]
			}
			sum += d*s + m*fsum
			continue
		}
		for i := 0; i < rem; i++ {
			qb := q[bo+4+(i>>1)]
			var qv float32
			if i&1 == 0 {
				qv = float32(qb & 0xF)
			} else {
				qv = float32(qb >> 4)
			}
			sum += (d*qv + m) * f[start+i]
		}
	}
	return sum, nil
}

func dotQ5_0F32AVX2(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q50BlockElems - 1) / q50BlockElems
	if len(q) < nBlocks*q50BlockBytes {
		return 0, fmt.Errorf("%w: dotQ5_0F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q50BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		qh := uint32(q[bo+2]) | uint32(q[bo+3])<<8 | uint32(q[bo+4])<<16 | uint32(q[bo+5])<<24
		start := b * 32
		rem := n - start
		if rem >= 32 {
			var tmp [32]int8
			for i := 0; i < 16; i++ {
				qb := q[bo+6+i]
				tmp[i*2] = int8((qb & 0x0F) | byte((qh>>uint(i*2))&1)<<4)
				tmp[i*2] -= 16
				tmp[i*2+1] = int8((qb >> 4) | byte((qh>>uint(i*2+1))&1)<<4)
				tmp[i*2+1] -= 16
			}
			fptr := (*float32)(unsafe.Pointer(&f[start]))
			sum += d * dotInt8F32x32AVX2Asm(&tmp[0], fptr)
			continue
		}
		for i := 0; i < rem; i++ {
			qb := q[bo+6+(i>>1)]
			hi := (qh >> uint(i)) & 1
			var low uint32
			if i&1 == 0 {
				low = uint32(qb & 0xF)
			} else {
				low = uint32(qb >> 4)
			}
			v := float32(int32(low|(hi<<4)) - 16)
			sum += d * v * f[start+i]
		}
	}
	return sum, nil
}

func dotQ5_1F32AVX2(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q51BlockElems - 1) / q51BlockElems
	if len(q) < nBlocks*q51BlockBytes {
		return 0, fmt.Errorf("%w: dotQ5_1F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q51BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		m := f16ToF32Local(uint16(q[bo+2]) | uint16(q[bo+3])<<8)
		qh := uint32(q[bo+4]) | uint32(q[bo+5])<<8 | uint32(q[bo+6])<<16 | uint32(q[bo+7])<<24
		start := b * 32
		rem := n - start
		if rem >= 32 {
			var tmp [32]int8
			for i := 0; i < 16; i++ {
				qb := q[bo+8+i]
				tmp[i*2] = int8((qb & 0x0F) | byte((qh>>uint(i*2))&1)<<4)
				tmp[i*2+1] = int8((qb >> 4) | byte((qh>>uint(i*2+1))&1)<<4)
			}
			fblk := f[start : start+32]
			fptr := (*float32)(unsafe.Pointer(&fblk[0]))
			s := dotInt8F32x32AVX2Asm(&tmp[0], fptr)
			var fsum float32
			for i := 0; i < 32; i++ {
				fsum += fblk[i]
			}
			sum += d*s + m*fsum
			continue
		}
		for i := 0; i < rem; i++ {
			qb := q[bo+8+(i>>1)]
			hi := (qh >> uint(i)) & 1
			var low uint32
			if i&1 == 0 {
				low = uint32(qb & 0xF)
			} else {
				low = uint32(qb >> 4)
			}
			v := float32(low | (hi << 4))
			sum += (d*v + m) * f[start+i]
		}
	}
	return sum, nil
}

func chooseAVX2DotKernel(quantType uint32, avx2Fn, genericFn func([]byte, []float32) (float32, error)) (func([]byte, []float32) (float32, error), string) {
	q, f, err := buildKernelSampleInputs(quantType, 384)
	if err != nil {
		return genericFn, "generic"
	}

	const iters = 128
	startA := time.Now()
	for i := 0; i < iters; i++ {
		_, _ = avx2Fn(q, f)
	}
	aDur := time.Since(startA)

	startG := time.Now()
	for i := 0; i < iters; i++ {
		_, _ = genericFn(q, f)
	}
	gDur := time.Since(startG)

	if aDur <= gDur {
		return avx2Fn, "avx2"
	}
	return genericFn, "generic"
}

func buildKernelSampleInputs(quantType uint32, cols int) ([]byte, []float32, error) {
	rowBytes, err := quantRowBytes(quantType, cols)
	if err != nil {
		return nil, nil, err
	}
	q := make([]byte, rowBytes)
	fillSyntheticQuantRowLocal(q, quantType)
	f := make([]float32, cols)
	rng := rand.New(rand.NewSource(42))
	for i := range f {
		f[i] = (rng.Float32() - 0.5) * 2
	}
	return q, f, nil
}
