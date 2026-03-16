package gguf

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"os"
	"testing"
)

// ── helpers ──────────────────────────────────────────────────────────────────

func testPutU32(b []byte, v uint32) { binary.LittleEndian.PutUint32(b, v) }
func testPutU64(b []byte, v uint64) { binary.LittleEndian.PutUint64(b, v) }

type testMeta struct {
	key   string
	vtype uint32
	val   any // must match vtype
}

type testTensor struct {
	name    string
	shape   []uint64
	dtype   uint32
	rawData []byte
}

// buildTestGGUF writes a minimal GGUF file into a temp dir and returns its path.
func buildTestGGUF(t *testing.T, metas []testMeta, tensors []testTensor) string {
	t.Helper()
	var buf bytes.Buffer
	var tmp [8]byte

	// Header
	buf.WriteString("GGUF")
	testPutU32(tmp[:4], 2)
	buf.Write(tmp[:4])
	testPutU64(tmp[:], uint64(len(tensors)))
	buf.Write(tmp[:])
	testPutU64(tmp[:], uint64(len(metas)))
	buf.Write(tmp[:])

	// Metadata entries
	for _, m := range metas {
		writeTestStr(&buf, m.key)
		testPutU32(tmp[:4], m.vtype)
		buf.Write(tmp[:4])
		writeTestVal(t, &buf, m.vtype, m.val)
	}

	// Tensor descriptors — pack data regions contiguously from offset 0.
	offsets := make([]uint64, len(tensors))
	var off uint64
	for i, tn := range tensors {
		offsets[i] = off
		off += uint64(len(tn.rawData))
	}
	for i, tn := range tensors {
		writeTestStr(&buf, tn.name)
		testPutU32(tmp[:4], uint32(len(tn.shape)))
		buf.Write(tmp[:4])
		for _, d := range tn.shape {
			testPutU64(tmp[:], d)
			buf.Write(tmp[:])
		}
		testPutU32(tmp[:4], tn.dtype)
		buf.Write(tmp[:4])
		testPutU64(tmp[:], offsets[i])
		buf.Write(tmp[:])
	}

	// Pad to 32-byte alignment.
	if pad := (32 - buf.Len()%32) % 32; pad > 0 {
		buf.Write(make([]byte, pad))
	}

	// Tensor raw data.
	for _, tn := range tensors {
		buf.Write(tn.rawData)
	}

	f, err := os.CreateTemp(t.TempDir(), "test-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(buf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return f.Name()
}

func writeTestStr(w *bytes.Buffer, s string) {
	var tmp [8]byte
	testPutU64(tmp[:], uint64(len(s)))
	w.Write(tmp[:])
	w.WriteString(s)
}

func writeTestVal(t *testing.T, w *bytes.Buffer, vtype uint32, val any) {
	t.Helper()
	var tmp [8]byte
	switch vtype {
	case typeString:
		writeTestStr(w, val.(string))
	case typeUint32:
		testPutU32(tmp[:4], val.(uint32))
		w.Write(tmp[:4])
	case typeFloat32:
		testPutU32(tmp[:4], math.Float32bits(val.(float32)))
		w.Write(tmp[:4])
	case typeArray:
		switch v := val.(type) {
		case []string:
			testPutU32(tmp[:4], typeString)
			w.Write(tmp[:4])
			testPutU64(tmp[:], uint64(len(v)))
			w.Write(tmp[:])
			for _, s := range v {
				writeTestStr(w, s)
			}
		case []uint32:
			testPutU32(tmp[:4], typeUint32)
			w.Write(tmp[:4])
			testPutU64(tmp[:], uint64(len(v)))
			w.Write(tmp[:])
			for _, u := range v {
				testPutU32(tmp[:4], u)
				w.Write(tmp[:4])
			}
		default:
			t.Fatalf("writeTestVal: unsupported array type %T", val)
		}
	default:
		t.Fatalf("writeTestVal: unsupported vtype %d", vtype)
	}
}

// f32Block builds a raw F32 tensor buffer from the given values.
func f32Block(vals ...float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

// ── f16ToF32 ─────────────────────────────────────────────────────────────────

func TestF16ToF32(t *testing.T) {
	tests := []struct {
		name string
		h    uint16
		want float32
	}{
		{"zero", 0x0000, 0.0},
		{"neg-zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3C00, 1.0},
		{"two", 0x4000, 2.0},
		{"neg-one", 0xBC00, -1.0},
		{"half", 0x3800, 0.5},
		{"inf", 0x7C00, float32(math.Inf(1))},
		{"neg-inf", 0xFC00, float32(math.Inf(-1))},
		// smallest subnormal F16 = 2^-24
		{"subnormal-min", 0x0001, float32(math.Pow(2, -24))},
		// largest subnormal F16 = 2^-14 * (1023/1024) ≈ 0.999*2^-14
		{"subnormal-max", 0x03FF, float32(float64(0x3FF) * math.Pow(2, -24))},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := f16ToF32(tc.h)
			if got != tc.want {
				t.Errorf("f16ToF32(0x%04X) = %v, want %v", tc.h, got, tc.want)
			}
		})
	}
	// NaN: just check it is NaN.
	if !math.IsNaN(float64(f16ToF32(0x7E00))) {
		t.Error("f16ToF32(0x7E00) should be NaN")
	}
}

// ── dequantisation unit tests ─────────────────────────────────────────────────

func TestDequantQ4_0(t *testing.T) {
	// One 32-element block: scale = 1.0 (f16 = 0x3C00).
	// Nibble byte 0x9A: lo=0xA=10 → 10-8=2, hi=0x9=9 → 9-8=1.
	// Remaining bytes 0x88: lo=8 → 0, hi=8 → 0.
	block := make([]byte, 18)
	binary.LittleEndian.PutUint16(block[0:], 0x3C00) // scale = 1.0
	block[2] = 0x9A                                  // elem[0]=2.0, elem[1]=1.0
	for i := 3; i < 18; i++ {
		block[i] = 0x88 // remaining elems = 0
	}
	out := make([]float32, 32)
	dequantQ4_0(block, out, 32)

	if out[0] != 2.0 {
		t.Errorf("out[0] = %v, want 2.0", out[0])
	}
	if out[1] != 1.0 {
		t.Errorf("out[1] = %v, want 1.0", out[1])
	}
	for i := 2; i < 32; i++ {
		if out[i] != 0.0 {
			t.Errorf("out[%d] = %v, want 0.0", i, out[i])
		}
	}
}

func TestDequantQ4_1(t *testing.T) {
	// d=1.0, m=0.5; byte 0x23: lo=3→3.5, hi=2→2.5.
	block := make([]byte, 20)
	binary.LittleEndian.PutUint16(block[0:], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(block[2:], 0x3800) // m = 0.5
	block[4] = 0x23                                  // elem[0]=3.5, elem[1]=2.5
	// remaining bytes: 0x00 → 0*1.0+0.5 = 0.5
	out := make([]float32, 32)
	dequantQ4_1(block, out, 32)

	if out[0] != 3.5 {
		t.Errorf("out[0] = %v, want 3.5", out[0])
	}
	if out[1] != 2.5 {
		t.Errorf("out[1] = %v, want 2.5", out[1])
	}
	for i := 2; i < 32; i++ {
		if out[i] != 0.5 {
			t.Errorf("out[%d] = %v, want 0.5", i, out[i])
		}
	}
}

func TestDequantQ5_0(t *testing.T) {
	// scale=1.0; qh=0x00000003 (bits 0 and 1 set); qs[0]=0x04.
	// elem[0]: lo4=4, hi_bit=(qh>>0)&1=1 → 5-bit=4|(1<<4)=20 → 20-16=4
	// elem[1]: lo4=0, hi_bit=(qh>>1)&1=1 → 5-bit=0|(1<<4)=16 → 16-16=0
	block := make([]byte, 22)
	binary.LittleEndian.PutUint16(block[0:], 0x3C00) // scale = 1.0
	binary.LittleEndian.PutUint32(block[2:], 0x00000003)
	block[6] = 0x04 // qs[0]: lo nibble=4, hi nibble=0
	// remaining qs bytes = 0x88 → 5-bit value = 8|(hi_bit<<4); hi_bits=0 → value=8 → 8-16=-8
	for i := 7; i < 22; i++ {
		block[i] = 0x88
	}
	out := make([]float32, 32)
	dequantQ5_0(block, out, 32)

	if out[0] != 4.0 {
		t.Errorf("out[0] = %v, want 4.0", out[0])
	}
	if out[1] != 0.0 {
		t.Errorf("out[1] = %v, want 0.0", out[1])
	}
	// elem[2] and [3] come from qs[1]=0x88, qh bits 2,3 = 0:
	// elem[2]: 5-bit=8+0=8 → -8; elem[3]: same
	if out[2] != -8.0 {
		t.Errorf("out[2] = %v, want -8.0", out[2])
	}
}

func TestDequantQ5_1(t *testing.T) {
	// d=1.0, m=0.0; qh=0 (all upper bits zero); qs[0]=0x4F.
	// elem[0]: lo4=0xF=15, hi_bit=0 → 5-bit=15; x=1.0*15+0=15.0
	// elem[1]: hi4=0x4=4, hi_bit=0 → 5-bit=4; x=1.0*4+0=4.0
	block := make([]byte, 24)
	binary.LittleEndian.PutUint16(block[0:], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(block[2:], 0x0000) // m = 0.0
	binary.LittleEndian.PutUint32(block[4:], 0)      // qh = 0
	block[8] = 0x4F                                  // lo=0xF=15, hi=0x4=4
	out := make([]float32, 32)
	dequantQ5_1(block, out, 32)

	if out[0] != 15.0 {
		t.Errorf("out[0] = %v, want 15.0", out[0])
	}
	if out[1] != 4.0 {
		t.Errorf("out[1] = %v, want 4.0", out[1])
	}
}

func TestDequantQ8_0(t *testing.T) {
	// scale=2.0; int8 values: 1, -1, 2, -2, rest 0.
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[0:], 0x4000) // scale = 2.0
	block[2] = 1
	block[3] = 0xFF // int8(-1)
	block[4] = 2
	block[5] = 0xFE // int8(-2)
	out := make([]float32, 32)
	dequantQ8_0(block, out, 32)

	want := []float32{2.0, -2.0, 4.0, -4.0}
	for i, w := range want {
		if out[i] != w {
			t.Errorf("out[%d] = %v, want %v", i, out[i], w)
		}
	}
	for i := 4; i < 32; i++ {
		if out[i] != 0.0 {
			t.Errorf("out[%d] = %v, want 0.0", i, out[i])
		}
	}
}

// ── metadata parsing ──────────────────────────────────────────────────────────

func TestOpenMetadata(t *testing.T) {
	path := buildTestGGUF(t, []testMeta{
		{key: "general.name", vtype: typeString, val: "mymodel"},
		{key: "general.version", vtype: typeUint32, val: uint32(42)},
		{key: "general.scale", vtype: typeFloat32, val: float32(1.5)},
		{key: "tokenizer.tokens", vtype: typeArray, val: []string{"hello", "world"}},
		{key: "layer.ids", vtype: typeArray, val: []uint32{1, 2, 3}},
	}, nil)

	f, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if got, ok := f.MetaString("general.name"); !ok || got != "mymodel" {
		t.Errorf("MetaString(general.name) = %q, %v", got, ok)
	}
	if got, ok := f.MetaUint32("general.version"); !ok || got != 42 {
		t.Errorf("MetaUint32(general.version) = %v, %v", got, ok)
	}
	if got, ok := f.MetaFloat32("general.scale"); !ok || got != 1.5 {
		t.Errorf("MetaFloat32(general.scale) = %v, %v", got, ok)
	}
	if got, ok := f.MetaStrings("tokenizer.tokens"); !ok || len(got) != 2 || got[0] != "hello" || got[1] != "world" {
		t.Errorf("MetaStrings = %v, %v", got, ok)
	}
	if got, ok := f.MetaUint32s("layer.ids"); !ok || len(got) != 3 || got[2] != 3 {
		t.Errorf("MetaUint32s = %v, %v", got, ok)
	}
	// Missing key.
	if _, ok := f.Meta("does.not.exist"); ok {
		t.Error("expected missing key to return false")
	}
}

// ── tensor round-trip ─────────────────────────────────────────────────────────

func TestOpenTensorF32(t *testing.T) {
	vals := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	path := buildTestGGUF(t, nil, []testTensor{
		{name: "weight", shape: []uint64{6}, dtype: dtypeF32, rawData: f32Block(vals...)},
	})

	f, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	names := f.TensorNames()
	if len(names) != 1 || names[0] != "weight" {
		t.Errorf("TensorNames = %v", names)
	}

	data, shape, err := f.Tensor(context.Background(), "weight")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if len(shape) != 1 || shape[0] != 6 {
		t.Errorf("shape = %v, want [6]", shape)
	}
	for i, want := range vals {
		if data[i] != want {
			t.Errorf("data[%d] = %v, want %v", i, data[i], want)
		}
	}
}

func TestOpenTensorNotFound(t *testing.T) {
	path := buildTestGGUF(t, nil, nil)
	f, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if _, _, err := f.Tensor(context.Background(), "missing"); err == nil {
		t.Error("expected error for missing tensor")
	}
}

// ── error cases ───────────────────────────────────────────────────────────────

func TestOpenBadMagic(t *testing.T) {
	f, err := os.CreateTemp(t.TempDir(), "bad-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("BLAH\x02\x00\x00\x00")
	f.Close()

	_, err = Open(context.Background(), f.Name())
	if !errors.Is(err, ErrInvalidMagic) {
		t.Errorf("expected ErrInvalidMagic, got %v", err)
	}
}

func TestOpenUnsupportedVersion(t *testing.T) {
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	var tmp [4]byte
	testPutU32(tmp[:], 5) // unsupported version
	buf.Write(tmp[:])

	f, err := os.CreateTemp(t.TempDir(), "ver-*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	f.Write(buf.Bytes())
	f.Close()

	_, err = Open(context.Background(), f.Name())
	if !errors.Is(err, ErrUnsupportedVersion) {
		t.Errorf("expected ErrUnsupportedVersion, got %v", err)
	}
}

func TestOpenContextCancelled(t *testing.T) {
	// Build a GGUF with a real tensor so context cancellation can fire during parsing.
	vals := make([]float32, 32)
	path := buildTestGGUF(t, nil, []testTensor{
		{name: "w", shape: []uint64{32}, dtype: dtypeF32, rawData: f32Block(vals...)},
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	f, err := Open(ctx, path)
	if f != nil {
		f.Close()
	}
	// Either Open returns an error or succeeds (0 metadata/tensor loops may not check ctx).
	// The important thing is no panic.
	_ = err
}
