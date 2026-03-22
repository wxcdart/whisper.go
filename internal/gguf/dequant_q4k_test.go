package gguf

import (
	"encoding/binary"
	"testing"
)

// TestDequantQ4K_Synthetic builds a single synthetic Q4_K super-block and
// verifies Dequantize produces the expected float32 outputs.
func TestDequantQ4K_Synthetic(t *testing.T) {
	// Build a single 144-byte super-block
	blk := make([]byte, 144)

	// d = 1.0 (f16 0x3C00), dmin = 0
	binary.LittleEndian.PutUint16(blk[0:2], 0x3C00)
	binary.LittleEndian.PutUint16(blk[2:4], 0x0000)

	// scales: set sc for j=0..7 = [1,2,3,4,5,6,7,8], m = 0
	// For j<4, sc in scales[j], m in scales[j+4]
	scales := make([]byte, K_SCALE_SIZE)
	scales[0] = 1
	scales[1] = 2
	scales[2] = 3
	scales[3] = 4
	// m values zero
	scales[4] = 0
	scales[5] = 0
	scales[6] = 0
	scales[7] = 0
	// For j>=4, store sc in scales[8..11] low nibble
	scales[8] = 5
	scales[9] = 6
	scales[10] = 7
	scales[11] = 8
	copy(blk[4:4+K_SCALE_SIZE], scales)

	// qs: 128 bytes. Choose lower nibble = 1, upper nibble = 2
	qbyte := byte((2 << 4) | 1)
	for i := 0; i < 128; i++ {
		blk[16+i] = qbyte
	}

	// Run dequant
	out, err := Dequantize(blk, dtypeQ4_K, uint64(256))
	if err != nil {
		t.Fatalf("Dequantize error: %v", err)
	}
	if len(out) != 256 {
		t.Fatalf("expected 256 outputs, got %d", len(out))
	}

	// Compute expected values: per group g (0..3), two sub-blocks sc0,sc1
	sc := []int{1, 2, 3, 4, 5, 6, 7, 8}
	a := float32(1) // lower nibble
	b := float32(2) // upper nibble

	idx := 0
	is := 0
	for g := 0; g < 4; g++ {
		sc0 := float32(sc[is+0])
		sc1 := float32(sc[is+1])
		// first 32 outputs: lower nibble * sc0
		for l := 0; l < 32; l++ {
			if out[idx] != sc0*a {
				t.Fatalf("mismatch at idx %d: got %v want %v", idx, out[idx], sc0*a)
			}
			idx++
		}
		// next 32 outputs: upper nibble * sc1
		for l := 0; l < 32; l++ {
			if out[idx] != sc1*b {
				t.Fatalf("mismatch at idx %d: got %v want %v", idx, out[idx], sc1*b)
			}
			idx++
		}
		is += 2
	}
}
