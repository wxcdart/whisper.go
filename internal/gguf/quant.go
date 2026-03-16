package gguf

import (
	"math"
	"strings"
)

const blockSize = 32

// float32ToF16 converts a float32 to IEEE-754 half-precision (f16).
func float32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 31)
	exp := int(bits>>23) & 0xFF
	mant := bits & 0x7FFFFF

	switch {
	case exp == 255: // Inf or NaN
		if mant != 0 {
			return sign<<15 | 0x7E00 // quiet NaN
		}
		return sign<<15 | 0x7C00 // Inf

	case exp == 0: // float32 subnormal or zero → rounds to ±0 in f16
		return sign << 15
	}

	exp16 := exp - 127 + 15
	if exp16 >= 31 {
		return sign<<15 | 0x7C00 // overflow → Inf
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign << 15 // underflow → ±0
		}
		// f16 subnormal
		shift := uint(1 - exp16)
		mant16 := uint16((mant | 0x800000) >> (13 + shift))
		return sign<<15 | mant16
	}

	// Round mantissa from 23 bits to 10 bits (round half-up).
	mant16 := uint16(mant >> 13)
	if (mant>>12)&1 == 1 {
		mant16++
	}
	if mant16 >= 0x400 {
		mant16 = 0
		exp16++
		if exp16 >= 31 {
			return sign<<15 | 0x7C00
		}
	}
	return sign<<15 | uint16(exp16)<<10 | mant16
}

// f16ToFloat32 converts an IEEE-754 half-precision value to float32.
func f16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	var bits uint32
	switch {
	case exp == 0 && mant == 0:
		bits = sign << 31 // ±0
	case exp == 0:
		// f16 subnormal → normalise for float32
		m := mant
		e := uint32(0)
		for m&0x400 == 0 {
			m <<= 1
			e++
		}
		m &= 0x3FF
		// exp32 = 113 - e; value = (1 + m/1024) * 2^(exp32-127)
		bits = sign<<31 | (113-e)<<23 | m<<13
	case exp == 31: // Inf or NaN
		bits = sign<<31 | 0xFF<<23 | mant<<13
	default:
		bits = sign<<31 | (exp+127-15)<<23 | mant<<13
	}
	return math.Float32frombits(bits)
}

// QuantizeF16 converts each float32 to f16; output is 2 bytes per element.
func QuantizeF16(data []float32) []byte {
	out := make([]byte, len(data)*2)
	for i, v := range data {
		h := float32ToF16(v)
		out[2*i] = byte(h)
		out[2*i+1] = byte(h >> 8)
	}
	return out
}

// QuantizeQ4_0 quantises data in blocks of 32 using the Q4_0 scheme.
// Each block: f16(scale=absmax/7) + 16 bytes of nibbles (low nibble first).
func QuantizeQ4_0(data []float32) []byte {
	nBlocks := (len(data) + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*18)
	for b := 0; b < nBlocks; b++ {
		blk := blockSlice(data, b)
		absmax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > absmax {
				absmax = a
			}
		}
		scale := absmax / 7.0
		ob := out[b*18:]
		putF16(ob[0:], scale)
		for i := 0; i < 16; i++ {
			lo := quantNibble(blk, 2*i, scale, 8)
			hi := quantNibble(blk, 2*i+1, scale, 8)
			ob[2+i] = lo | (hi << 4)
		}
	}
	return out
}

// QuantizeQ4_1 quantises data in blocks of 32 using the Q4_1 scheme.
// Each block: f16(scale) + f16(offset=min) + 16 bytes of nibbles.
func QuantizeQ4_1(data []float32) []byte {
	nBlocks := (len(data) + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*20)
	for b := 0; b < nBlocks; b++ {
		blk := blockSlice(data, b)
		mn, mx := blockMinMax(blk)
		scale := (mx - mn) / 15.0
		ob := out[b*20:]
		putF16(ob[0:], scale)
		putF16(ob[2:], mn)
		for i := 0; i < 16; i++ {
			lo := quantNibbleUnsigned(blk, 2*i, scale, mn)
			hi := quantNibbleUnsigned(blk, 2*i+1, scale, mn)
			ob[4+i] = lo | (hi << 4)
		}
	}
	return out
}

// QuantizeQ5_0 quantises data in blocks of 32 using the Q5_0 scheme.
// Each block: f16(scale=absmax/15) + uint32(qh) + 16 bytes of lower 4 bits.
func QuantizeQ5_0(data []float32) []byte {
	nBlocks := (len(data) + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*22)
	for b := 0; b < nBlocks; b++ {
		blk := blockSlice(data, b)
		absmax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > absmax {
				absmax = a
			}
		}
		scale := absmax / 15.0
		ob := out[b*22:]
		putF16(ob[0:], scale)
		var qh uint32
		for i := 0; i < blockSize; i++ {
			q := quant5Signed(blk, i, scale, 16)
			if (q>>4)&1 == 1 {
				qh |= 1 << uint(i)
			}
			nibbleSet(ob[6:], i, uint8(q&0xF))
		}
		ob[2] = byte(qh)
		ob[3] = byte(qh >> 8)
		ob[4] = byte(qh >> 16)
		ob[5] = byte(qh >> 24)
	}
	return out
}

// QuantizeQ5_1 quantises data in blocks of 32 using the Q5_1 scheme.
// Each block: f16(scale) + f16(min) + uint32(qh) + 16 bytes of lower 4 bits.
func QuantizeQ5_1(data []float32) []byte {
	nBlocks := (len(data) + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*24)
	for b := 0; b < nBlocks; b++ {
		blk := blockSlice(data, b)
		mn, mx := blockMinMax(blk)
		scale := (mx - mn) / 31.0
		ob := out[b*24:]
		putF16(ob[0:], scale)
		putF16(ob[2:], mn)
		var qh uint32
		for i := 0; i < blockSize; i++ {
			q := quant5Unsigned(blk, i, scale, mn)
			if (q>>4)&1 == 1 {
				qh |= 1 << uint(i)
			}
			nibbleSet(ob[8:], i, uint8(q&0xF))
		}
		ob[4] = byte(qh)
		ob[5] = byte(qh >> 8)
		ob[6] = byte(qh >> 16)
		ob[7] = byte(qh >> 24)
	}
	return out
}

// QuantizeQ8_0 quantises data in blocks of 32 using the Q8_0 scheme.
// Each block: f16(scale=absmax/127) + 32 bytes of int8.
func QuantizeQ8_0(data []float32) []byte {
	nBlocks := (len(data) + blockSize - 1) / blockSize
	out := make([]byte, nBlocks*34)
	for b := 0; b < nBlocks; b++ {
		blk := blockSlice(data, b)
		absmax := float32(0)
		for _, v := range blk {
			if a := float32(math.Abs(float64(v))); a > absmax {
				absmax = a
			}
		}
		scale := absmax / 127.0
		ob := out[b*34:]
		putF16(ob[0:], scale)
		for i := 0; i < blockSize; i++ {
			v := float32(0)
			if i < len(blk) {
				v = blk[i]
			}
			q := int8(0)
			if scale != 0 {
				q = clampI8(int32(math.Round(float64(v / scale))))
			}
			ob[2+i] = byte(q)
		}
	}
	return out
}

// shouldQuantize returns true for weight matrices that benefit from quantisation.
// Biases, layer norms, positional embeddings, and conv biases are kept F32.
func shouldQuantize(name string) bool {
	lower := strings.ToLower(name)
	// Keep norms and layer-norm weights as F32.
	if strings.Contains(lower, "norm") || strings.Contains(lower, ".ln") {
		return false
	}
	// Keep biases as F32.
	if strings.HasSuffix(lower, ".bias") {
		return false
	}
	// Quantise weight matrices.
	return strings.HasSuffix(lower, ".weight")
}

// --- helpers ----------------------------------------------------------------

func blockSlice(data []float32, b int) []float32 {
	start := b * blockSize
	end := start + blockSize
	if end > len(data) {
		end = len(data)
	}
	return data[start:end]
}

func blockMinMax(blk []float32) (mn, mx float32) {
	if len(blk) == 0 {
		return 0, 0
	}
	mn, mx = blk[0], blk[0]
	for _, v := range blk[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	return mn, mx
}

func putF16(dst []byte, f float32) {
	h := float32ToF16(f)
	dst[0] = byte(h)
	dst[1] = byte(h >> 8)
}

// quantNibble returns a 4-bit signed value (biased by bias) for data[i]/scale.
func quantNibble(blk []float32, i int, scale float32, bias int32) uint8 {
	v := float32(0)
	if i < len(blk) {
		v = blk[i]
	}
	q := bias
	if scale != 0 {
		q = int32(math.Round(float64(v/scale))) + bias
	}
	if q < 0 {
		q = 0
	}
	if q > 15 {
		q = 15
	}
	return uint8(q)
}

// quantNibbleUnsigned returns a 4-bit unsigned value for (data[i]-mn)/scale.
func quantNibbleUnsigned(blk []float32, i int, scale, mn float32) uint8 {
	v := float32(0)
	if i < len(blk) {
		v = blk[i]
	}
	q := int32(0)
	if scale != 0 {
		q = int32(math.Round(float64((v - mn) / scale)))
	}
	if q < 0 {
		q = 0
	}
	if q > 15 {
		q = 15
	}
	return uint8(q)
}

// quant5Signed returns a 5-bit value (biased by bias) for data[i]/scale.
func quant5Signed(blk []float32, i int, scale float32, bias int32) uint8 {
	v := float32(0)
	if i < len(blk) {
		v = blk[i]
	}
	q := bias
	if scale != 0 {
		q = int32(math.Round(float64(v/scale))) + bias
	}
	if q < 0 {
		q = 0
	}
	if q > 31 {
		q = 31
	}
	return uint8(q)
}

// quant5Unsigned returns a 5-bit unsigned value for (data[i]-mn)/scale.
func quant5Unsigned(blk []float32, i int, scale, mn float32) uint8 {
	v := float32(0)
	if i < len(blk) {
		v = blk[i]
	}
	q := int32(0)
	if scale != 0 {
		q = int32(math.Round(float64((v - mn) / scale)))
	}
	if q < 0 {
		q = 0
	}
	if q > 31 {
		q = 31
	}
	return uint8(q)
}

// nibbleSet stores a 4-bit value into the nibble array at position i (low nibble of byte i/2).
func nibbleSet(dst []byte, i int, v uint8) {
	if i%2 == 0 {
		dst[i/2] = (dst[i/2] & 0xF0) | (v & 0xF)
	} else {
		dst[i/2] = (dst[i/2] & 0x0F) | ((v & 0xF) << 4)
	}
}

func clampI8(v int32) int8 {
	if v < -127 {
		return -127
	}
	if v > 127 {
		return 127
	}
	return int8(v)
}
