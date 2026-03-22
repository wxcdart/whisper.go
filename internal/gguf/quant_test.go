package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestQuantRoundTrip verifies that quantise → dequantise stays within the
// expected max absolute error for each quantisation type.
func TestQuantRoundTrip(t *testing.T) {
	t.Parallel()

	// 32 representative values spread over [-1, 1].
	input := make([]float32, blockSize)
	for i := range input {
		input[i] = float32(i)/float32(blockSize-1)*2 - 1 // -1 … +1
	}

	cases := []struct {
		name   string
		encode func([]float32) []byte
		decode func([]byte) []float32
		maxErr float32
	}{
		{
			name:   "Q4_0",
			encode: QuantizeQ4_0,
			decode: testDequantQ4_0,
			maxErr: 0.1,
		},
		{
			name:   "Q4_1",
			encode: QuantizeQ4_1,
			decode: testDequantQ4_1,
			maxErr: 0.1,
		},
		{
			name:   "Q5_0",
			encode: QuantizeQ5_0,
			decode: testDequantQ5_0,
			maxErr: 0.05,
		},
		{
			name:   "Q5_1",
			encode: QuantizeQ5_1,
			decode: testDequantQ5_1,
			maxErr: 0.05,
		},
		{
			name:   "Q8_0",
			encode: QuantizeQ8_0,
			decode: testDequantQ8_0,
			maxErr: 0.01,
		},
		{
			name:   "F16",
			encode: QuantizeF16,
			decode: dequantF16,
			maxErr: 0.001,
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			encoded := tc.encode(input)
			decoded := tc.decode(encoded)
			if len(decoded) < len(input) {
				t.Fatalf("decoded length %d < input length %d", len(decoded), len(input))
			}
			var maxErr float32
			for i, orig := range input {
				diff := float32(math.Abs(float64(orig - decoded[i])))
				if diff > maxErr {
					maxErr = diff
				}
			}
			if maxErr > tc.maxErr {
				t.Errorf("max absolute error %.6f exceeds tolerance %.6f", maxErr, tc.maxErr)
			}
		})
	}
}

// TestF16Conversion checks specific float32 ↔ f16 round-trips.
func TestF16Conversion(t *testing.T) {
	t.Parallel()

	// exact 2^-24 as float32 (smallest positive f16 subnormal)
	minF16Sub := math.Float32frombits(0x33800000) // = 2^-24

	cases := []struct {
		name   string
		f32    float32
		maxErr float32
	}{
		{"zero", 0.0, 0},
		{"one", 1.0, 0},
		{"neg_one", -1.0, 0},
		{"half", 0.5, 0},
		{"max_f16", 65504.0, 0},
		{"min_f16_subnormal", minF16Sub, 1e-30},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			h := float32ToF16(tc.f32)
			got := f16ToFloat32(h)
			diff := float32(math.Abs(float64(tc.f32 - got)))
			tol := tc.maxErr
			if tol == 0 {
				tol = float32(math.Abs(float64(tc.f32))) * 1e-3
				if tol < 1e-6 {
					tol = 1e-6
				}
			}
			if diff > tol {
				t.Errorf("float32ToF16/f16ToFloat32(%v): got %v, diff %v (tol %v)",
					tc.f32, got, diff, tol)
			}
		})
	}
}

// --- inline dequantisation helpers for tests --------------------------------

func dequantF16(data []byte) []float32 {
	n := len(data) / 2
	out := make([]float32, n)
	for i := range out {
		h := binary.LittleEndian.Uint16(data[2*i:])
		out[i] = f16ToFloat32(h)
	}
	return out
}

func testDequantQ4_0(data []byte) []float32 {
	const stride = 18
	nBlocks := len(data) / stride
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		ob := data[b*stride:]
		scale := f16ToFloat32(binary.LittleEndian.Uint16(ob[0:]))
		for i := 0; i < blockSize; i++ {
			nibble := uint8(0)
			if i%2 == 0 {
				nibble = ob[2+i/2] & 0xF
			} else {
				nibble = (ob[2+i/2] >> 4) & 0xF
			}
			out[b*blockSize+i] = (float32(nibble) - 8) * scale
		}
	}
	return out
}

func testDequantQ4_1(data []byte) []float32 {
	const stride = 20
	nBlocks := len(data) / stride
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		ob := data[b*stride:]
		scale := f16ToFloat32(binary.LittleEndian.Uint16(ob[0:]))
		offset := f16ToFloat32(binary.LittleEndian.Uint16(ob[2:]))
		for i := 0; i < blockSize; i++ {
			nibble := uint8(0)
			if i%2 == 0 {
				nibble = ob[4+i/2] & 0xF
			} else {
				nibble = (ob[4+i/2] >> 4) & 0xF
			}
			out[b*blockSize+i] = float32(nibble)*scale + offset
		}
	}
	return out
}

func testDequantQ5_0(data []byte) []float32 {
	const stride = 22
	nBlocks := len(data) / stride
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		ob := data[b*stride:]
		scale := f16ToFloat32(binary.LittleEndian.Uint16(ob[0:]))
		qh := binary.LittleEndian.Uint32(ob[2:])
		for i := 0; i < blockSize; i++ {
			lo := uint8(0)
			if i%2 == 0 {
				lo = ob[6+i/2] & 0xF
			} else {
				lo = (ob[6+i/2] >> 4) & 0xF
			}
			bit4 := uint8((qh >> uint(i)) & 1)
			q5 := int32(bit4<<4|lo) - 16
			out[b*blockSize+i] = float32(q5) * scale
		}
	}
	return out
}

func testDequantQ5_1(data []byte) []float32 {
	const stride = 24
	nBlocks := len(data) / stride
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		ob := data[b*stride:]
		scale := f16ToFloat32(binary.LittleEndian.Uint16(ob[0:]))
		mn := f16ToFloat32(binary.LittleEndian.Uint16(ob[2:]))
		qh := binary.LittleEndian.Uint32(ob[4:])
		for i := 0; i < blockSize; i++ {
			lo := uint8(0)
			if i%2 == 0 {
				lo = ob[8+i/2] & 0xF
			} else {
				lo = (ob[8+i/2] >> 4) & 0xF
			}
			bit4 := uint8((qh >> uint(i)) & 1)
			q5 := bit4<<4 | lo
			out[b*blockSize+i] = float32(q5)*scale + mn
		}
	}
	return out
}

func testDequantQ8_0(data []byte) []float32 {
	const stride = 34
	nBlocks := len(data) / stride
	out := make([]float32, nBlocks*blockSize)
	for b := 0; b < nBlocks; b++ {
		ob := data[b*stride:]
		scale := f16ToFloat32(binary.LittleEndian.Uint16(ob[0:]))
		for i := 0; i < blockSize; i++ {
			out[b*blockSize+i] = float32(int8(ob[2+i])) * scale
		}
	}
	return out
}
