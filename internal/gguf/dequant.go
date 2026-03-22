package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	QK_K         = 256
	K_SCALE_SIZE = 12
)

// f16ToF32 converts an IEEE 754 half-precision value to float32.
func f16ToF32(h uint16) float32 {
	s := uint32(h>>15) << 31
	e := uint32((h >> 10) & 0x1F)
	m := uint32(h & 0x3FF)

	switch e {
	case 0:
		if m == 0 {
			return math.Float32frombits(s)
		}
		// Subnormal F16 → normalised F32.
		// Shift mantissa left until the implicit leading bit appears at bit 10,
		// tracking the number of shifts to adjust the exponent.
		k := uint32(1)
		m <<= 1
		for m&0x400 == 0 {
			m <<= 1
			k++
		}
		m &= 0x3FF
		// F32 biased exponent: 127 - 15 - k + 1 = 113 - k
		return math.Float32frombits(s | ((113 - k) << 23) | (m << 13))
	case 31:
		// Infinity or NaN.
		return math.Float32frombits(s | 0x7F800000 | (m << 13))
	default:
		// Normal: re-bias exponent from 15 to 127 (diff = 112).
		return math.Float32frombits(s | ((e + 112) << 23) | (m << 13))
	}
}

// dequantize converts raw quantised bytes to float32 values.
func dequantize(raw []byte, dtype uint32, numElems uint64) ([]float32, error) {
	out := make([]float32, numElems)
	n := int(numElems)
	switch dtype {
	case dtypeF32:
		if uint64(len(raw)) < numElems*4 {
			return nil, fmt.Errorf("short F32 buffer")
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case dtypeF16:
		if uint64(len(raw)) < numElems*2 {
			return nil, fmt.Errorf("short F16 buffer")
		}
		for i := 0; i < n; i++ {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case dtypeQ4_0:
		dequantQ4_0(raw, out, n)
	case dtypeQ4_1:
		dequantQ4_1(raw, out, n)
	case dtypeQ5_0:
		dequantQ5_0(raw, out, n)
	case dtypeQ5_1:
		dequantQ5_1(raw, out, n)
	case dtypeQ8_0:
		dequantQ8_0(raw, out, n)
	case dtypeQ4_K:
		dequantQ4_K(raw, out, n)
	default:
		return nil, fmt.Errorf("unsupported dtype: %d", dtype)
	}
	return out, nil
}

// Dequantize converts raw quantised bytes to float32 values. Exported for
// use by alternative file format adapters (e.g. legacy ggml .bin parser).
func Dequantize(raw []byte, dtype uint32, numElems uint64) ([]float32, error) {
	return dequantize(raw, dtype, numElems)
}

// dequantQ4_0 decodes Q4_0 blocks.
// Block layout (18 bytes): [0:2] f16 scale | [2:18] 16 bytes of 32 nibbles.
// x[i] = scale * (nibble[i] - 8)
func dequantQ4_0(raw []byte, out []float32, n int) {
	nb := (n + 31) / 32
	for b := 0; b < nb; b++ {
		block := raw[b*18:]
		scale := f16ToF32(binary.LittleEndian.Uint16(block))
		qs := block[2:]
		base := b * 32
		for i := 0; i < 16; i++ {
			if j := base + i*2; j < n {
				out[j] = scale * float32(int32(qs[i]&0xF)-8)
			}
			if j := base + i*2 + 1; j < n {
				out[j] = scale * float32(int32(qs[i]>>4)-8)
			}
		}
	}
}

// dequantQ4_1 decodes Q4_1 blocks.
// Block layout (20 bytes): [0:2] f16 d | [2:4] f16 m | [4:20] 16 bytes of 32 nibbles.
// x[i] = d * nibble[i] + m
func dequantQ4_1(raw []byte, out []float32, n int) {
	nb := (n + 31) / 32
	for b := 0; b < nb; b++ {
		block := raw[b*20:]
		d := f16ToF32(binary.LittleEndian.Uint16(block))
		m := f16ToF32(binary.LittleEndian.Uint16(block[2:]))
		qs := block[4:]
		base := b * 32
		for i := 0; i < 16; i++ {
			if j := base + i*2; j < n {
				out[j] = d*float32(qs[i]&0xF) + m
			}
			if j := base + i*2 + 1; j < n {
				out[j] = d*float32(qs[i]>>4) + m
			}
		}
	}
}

// dequantQ5_0 decodes Q5_0 blocks.
// Block layout (22 bytes): [0:2] f16 scale | [2:6] uint32 qh | [6:22] 16 bytes qs.
// 5-bit value = (lower4 nibble) | ((qh >> j) & 1) << 4; x[i] = scale * (value - 16)
func dequantQ5_0(raw []byte, out []float32, n int) {
	nb := (n + 31) / 32
	for b := 0; b < nb; b++ {
		block := raw[b*22:]
		scale := f16ToF32(binary.LittleEndian.Uint16(block))
		qh := binary.LittleEndian.Uint32(block[2:])
		qs := block[6:]
		base := b * 32
		for i := 0; i < 16; i++ {
			hi0 := (qh >> uint(i*2)) & 1
			hi1 := (qh >> uint(i*2+1)) & 1
			v0 := int32(uint32(qs[i]&0xF)|(hi0<<4)) - 16
			v1 := int32(uint32(qs[i]>>4)|(hi1<<4)) - 16
			if j := base + i*2; j < n {
				out[j] = scale * float32(v0)
			}
			if j := base + i*2 + 1; j < n {
				out[j] = scale * float32(v1)
			}
		}
	}
}

// dequantQ5_1 decodes Q5_1 blocks.
// Block layout (24 bytes): [0:2] f16 d | [2:4] f16 m | [4:8] uint32 qh | [8:24] 16 bytes qs.
// 5-bit value = (lower4 nibble) | ((qh >> j) & 1) << 4; x[i] = d * value + m
func dequantQ5_1(raw []byte, out []float32, n int) {
	nb := (n + 31) / 32
	for b := 0; b < nb; b++ {
		block := raw[b*24:]
		d := f16ToF32(binary.LittleEndian.Uint16(block))
		m := f16ToF32(binary.LittleEndian.Uint16(block[2:]))
		qh := binary.LittleEndian.Uint32(block[4:])
		qs := block[8:]
		base := b * 32
		for i := 0; i < 16; i++ {
			hi0 := (qh >> uint(i*2)) & 1
			hi1 := (qh >> uint(i*2+1)) & 1
			v0 := float32(uint32(qs[i]&0xF) | (hi0 << 4))
			v1 := float32(uint32(qs[i]>>4) | (hi1 << 4))
			if j := base + i*2; j < n {
				out[j] = d*v0 + m
			}
			if j := base + i*2 + 1; j < n {
				out[j] = d*v1 + m
			}
		}
	}
}

// dequantQ8_0 decodes Q8_0 blocks.
// Block layout (34 bytes): [0:2] f16 scale | [2:34] 32 int8 values.
// x[i] = scale * int8[i]
func dequantQ8_0(raw []byte, out []float32, n int) {
	nb := (n + 31) / 32
	for b := 0; b < nb; b++ {
		block := raw[b*34:]
		scale := f16ToF32(binary.LittleEndian.Uint16(block))
		qs := block[2:]
		base := b * 32
		for i := 0; i < 32; i++ {
			if j := base + i; j < n {
				out[j] = scale * float32(int8(qs[i]))
			}
		}
	}
}

// dequantQ4_K decodes Q4_K blocks (k-quant 4-bit).
// dequantQ4_K decodes Q4_K blocks (k-quant 4-bit).
// Ported from ggml's `dequantize_row_q4_K` implementation.
// Super-block layout (144 bytes per 256 elements):
// [0:2] f16 d | [2:4] f16 dmin | [4:16] 12 bytes scales | [16:144] 128 bytes qs
func dequantQ4_K(raw []byte, out []float32, n int) {
	const (
		superElems  = 256
		superBytes  = 144
		scalesStart = 4
		qsStart     = 16
	)

	nb := (n + superElems - 1) / superElems
	outIdx := 0

	getScaleMinK4 := func(j int, scales []byte) (uint8, uint8) {
		if j < 4 {
			d := scales[j] & 63
			m := scales[j+4] & 63
			return d, m
		}
		d := (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
		m := (scales[j+4] >> 4) | ((scales[j-0] >> 6) << 4)
		return d, m
	}

	for b := 0; b < nb; b++ {
		base := b * superBytes
		if base+qsStart > len(raw) {
			break
		}
		blk := raw[base:]

		d := f16ToF32(binary.LittleEndian.Uint16(blk[0:2]))
		dmin := f16ToF32(binary.LittleEndian.Uint16(blk[2:4]))
		scales := blk[scalesStart : scalesStart+K_SCALE_SIZE]
		qs := blk[qsStart : qsStart+QK_K/2]

		is := 0
		// There are 4 groups of 64 elements; each group consumes 32 bytes from qs
		for g := 0; g < 4; g++ {
			// two sub-blocks per group
			sc0, m0 := getScaleMinK4(is+0, scales)
			sc1, m1 := getScaleMinK4(is+1, scales)
			d1 := d * float32(sc0)
			m1f := dmin * float32(m0)
			d2 := d * float32(sc1)
			m2f := dmin * float32(m1)

			qoff := g * 32
			// lower nibble values (first 32 outputs)
			for l := 0; l < 32; l++ {
				if outIdx >= n {
					return
				}
				v := qs[qoff+l] & 0xF
				out[outIdx] = d1*float32(v) - m1f
				outIdx++
			}
			// upper nibble values (next 32 outputs)
			for l := 0; l < 32; l++ {
				if outIdx >= n {
					return
				}
				v := qs[qoff+l] >> 4
				out[outIdx] = d2*float32(v) - m2f
				outIdx++
			}

			is += 2
		}
	}
}
