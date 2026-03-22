//go:build arm64

package ml

import "fmt"

func dotQ8_0F32NEON(q []byte, f []float32) (float32, error) {
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
		if rem >= 32 {
			sum += dotQ8Block32(scale, q[bo+2:bo+2+q80BlockElems], f[start:start+q80BlockElems])
			continue
		}
		for i := 0; i < rem; i++ {
			sum += scale * float32(int8(q[bo+2+i])) * f[start+i]
		}
	}
	return sum, nil
}

func dotQ4_0F32NEON(q []byte, f []float32) (float32, error) {
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
			sum += dotQ40Block32(d, q[bo+2:bo+18], f[start:start+32])
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

func dotQ4_1F32NEON(q []byte, f []float32) (float32, error) {
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
			sum += dotQ41Block32(d, m, q[bo+4:bo+20], f[start:start+32])
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

func dotQ5_0F32NEON(q []byte, f []float32) (float32, error) {
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
			sum += dotQ50Block32(d, qh, q[bo+6:bo+22], f[start:start+32])
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

func dotQ5_1F32NEON(q []byte, f []float32) (float32, error) {
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
			sum += dotQ51Block32(d, m, qh, q[bo+8:bo+24], f[start:start+32])
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
