package gguf

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

const (
	ggufMagicU32 = uint32(0x46554747) // 'GGUF' in little-endian
	ggufVersion  = uint32(3)
	ggufAlign    = int64(32)
)

// WriteFile serialises f to a GGUF v3 file at path.
func WriteFile(ctx context.Context, path string, f *WritableFile) error {
	return writeGGUF(ctx, path, f)
}

// writeGGUF is the internal GGUF v3 serialiser.
func writeGGUF(_ context.Context, path string, f *WritableFile) error {
	// Pre-compute header size so we can calculate the data section alignment
	// and tensor offsets before we start writing.
	headerSize := int64(4 + 4 + 8 + 8) // magic + version + tensor_count + meta_count
	for i := range f.meta {
		headerSize += ggufStringSize(f.meta[i].key)
		headerSize += 4 // value_type uint32
		headerSize += ggufValueSize(f.meta[i].vtype, f.meta[i].val)
	}
	for _, t := range f.tensors {
		headerSize += ggufStringSize(t.name) // name
		headerSize += 4                      // n_dimensions uint32
		headerSize += int64(len(t.shape)) * 8 // dimensions uint64 each
		headerSize += 4                      // type uint32
		headerSize += 8                      // offset uint64
	}

	// Pad header to 32-byte alignment.
	padding := int64(0)
	if headerSize%ggufAlign != 0 {
		padding = ggufAlign - (headerSize % ggufAlign)
	}

	// Compute per-tensor data offsets (relative to data section start).
	offsets := make([]uint64, len(f.tensors))
	off := uint64(0)
	for i, t := range f.tensors {
		offsets[i] = off
		off += uint64(len(t.data))
		if uint64Mod32(off) != 0 {
			off += 32 - uint64Mod32(off)
		}
	}

	// Create output file.
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("gguf: write: %w", err)
	}
	defer file.Close() //nolint:errcheck

	bw := bufio.NewWriterSize(file, 1<<20)

	// --- Header ---
	putU32(bw, ggufMagicU32)
	putU32(bw, ggufVersion)
	putU64(bw, uint64(len(f.tensors)))
	putU64(bw, uint64(len(f.meta)))

	// --- Metadata ---
	for _, kv := range f.meta {
		writeString(bw, kv.key)
		putU32(bw, uint32(kv.vtype))
		writeMetaValue(bw, kv.vtype, kv.val)
	}

	// --- Tensor info ---
	for i, t := range f.tensors {
		writeString(bw, t.name)
		putU32(bw, uint32(len(t.shape)))
		for _, dim := range t.shape {
			putU64(bw, dim)
		}
		putU32(bw, uint32(t.dtype))
		putU64(bw, offsets[i])
	}

	// --- Alignment padding ---
	bw.Write(make([]byte, padding)) //nolint:errcheck

	// --- Tensor data ---
	for i, t := range f.tensors {
		bw.Write(t.data) //nolint:errcheck
		// Pad after each tensor except the last.
		if i < len(f.tensors)-1 {
			end := offsets[i] + uint64(len(t.data))
			if p := uint64Mod32(end); p != 0 {
				bw.Write(make([]byte, 32-p)) //nolint:errcheck
			}
		}
	}

	return bw.Flush()
}

// ggufStringSize returns the number of bytes a GGUF string occupies on disk.
func ggufStringSize(s string) int64 {
	return 8 + int64(len(s)) // uint64 length + bytes
}

// ggufValueSize returns the number of bytes a metadata value occupies on disk.
func ggufValueSize(vtype metaType, val any) int64 {
	switch vtype {
	case metaUint8, metaInt8, metaBool:
		return 1
	case metaUint16, metaInt16:
		return 2
	case metaUint32, metaInt32, metaFloat32:
		return 4
	case metaUint64, metaInt64, metaFloat64:
		return 8
	case metaString:
		if s, ok := val.(string); ok {
			return ggufStringSize(s)
		}
		return 8
	case metaArray:
		if av, ok := val.(arrayVal); ok {
			total := int64(4 + 8) // elem_type + count
			for _, elem := range av.elems {
				total += ggufValueSize(av.elemType, elem)
			}
			return total
		}
		return 4 + 8
	default:
		return 0
	}
}

// writeMetaValue writes the binary encoding of a metadata value.
func writeMetaValue(bw *bufio.Writer, vtype metaType, val any) {
	switch vtype {
	case metaUint8:
		bw.WriteByte(valUint8(val)) //nolint:errcheck
	case metaInt8:
		bw.WriteByte(byte(valInt8(val))) //nolint:errcheck
	case metaUint16:
		putU16(bw, valUint16(val))
	case metaInt16:
		putU16(bw, uint16(valInt16(val)))
	case metaUint32:
		putU32(bw, valUint32(val))
	case metaInt32:
		putU32(bw, uint32(valInt32(val)))
	case metaFloat32:
		putU32(bw, math.Float32bits(valFloat32(val)))
	case metaBool:
		b := byte(0)
		if valBool(val) {
			b = 1
		}
		bw.WriteByte(b) //nolint:errcheck
	case metaString:
		writeString(bw, valString(val))
	case metaUint64:
		putU64(bw, valUint64(val))
	case metaInt64:
		putU64(bw, uint64(valInt64(val)))
	case metaFloat64:
		putU64(bw, math.Float64bits(valFloat64(val)))
	case metaArray:
		if av, ok := val.(arrayVal); ok {
			putU32(bw, uint32(av.elemType))
			putU64(bw, uint64(len(av.elems)))
			for _, elem := range av.elems {
				writeMetaValue(bw, av.elemType, elem)
			}
		}
	}
}

// writeString writes a GGUF string (uint64 length + raw bytes, no null terminator).
func writeString(bw *bufio.Writer, s string) {
	putU64(bw, uint64(len(s)))
	bw.WriteString(s) //nolint:errcheck
}

func putU16(bw *bufio.Writer, v uint16) {
	binary.Write(bw, binary.LittleEndian, v) //nolint:errcheck
}

func putU32(bw *bufio.Writer, v uint32) {
	binary.Write(bw, binary.LittleEndian, v) //nolint:errcheck
}

func putU64(bw *bufio.Writer, v uint64) {
	binary.Write(bw, binary.LittleEndian, v) //nolint:errcheck
}

func uint64Mod32(v uint64) uint64 { return v % 32 }

// --- type-assertion helpers (return zero values on type mismatch) ---

func valUint8(v any) uint8 {
	if x, ok := v.(uint8); ok {
		return x
	}
	return 0
}

func valInt8(v any) int8 {
	if x, ok := v.(int8); ok {
		return x
	}
	return 0
}

func valUint16(v any) uint16 {
	if x, ok := v.(uint16); ok {
		return x
	}
	return 0
}

func valInt16(v any) int16 {
	if x, ok := v.(int16); ok {
		return x
	}
	return 0
}

func valUint32(v any) uint32 {
	if x, ok := v.(uint32); ok {
		return x
	}
	return 0
}

func valInt32(v any) int32 {
	if x, ok := v.(int32); ok {
		return x
	}
	return 0
}

func valFloat32(v any) float32 {
	if x, ok := v.(float32); ok {
		return x
	}
	return 0
}

func valBool(v any) bool {
	if x, ok := v.(bool); ok {
		return x
	}
	return false
}

func valString(v any) string {
	if x, ok := v.(string); ok {
		return x
	}
	return ""
}

func valUint64(v any) uint64 {
	if x, ok := v.(uint64); ok {
		return x
	}
	return 0
}

func valInt64(v any) int64 {
	if x, ok := v.(int64); ok {
		return x
	}
	return 0
}

func valFloat64(v any) float64 {
	if x, ok := v.(float64); ok {
		return x
	}
	return 0
}
