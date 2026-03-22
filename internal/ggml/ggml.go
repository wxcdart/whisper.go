package ggml

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"

	"github.com/whispergo/whisper.go/internal/gguf"
)

var (
	ErrOldFormatNotImplemented = errors.New("internal/ggml: legacy ggml (.bin) format parsing not implemented")
)

// Open detects the model format and returns a gguf.File-compatible object.
// Currently GGUF files are handled by delegating to internal/gguf.Open.
// Old-style ggml `.bin` files (magic 'lmgg' / legacy) are detected but not
// parsed yet — the function returns a helpful error directing users to
// conversion tools or to add a native parser.
func Open(ctx context.Context, path string) (gguf.FileLike, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open model: %w", err)
	}
	defer f.Close()

	var mag [4]byte
	if _, err := io.ReadFull(f, mag[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}

	switch string(mag[:]) {
	case "GGUF":
		// Rewind and let internal/gguf parse normally.
		return gguf.Open(ctx, path)
	case "lmgg", "GGML":
		// Legacy ggml .bin format detected. Parse basic header and return
		// a lightweight adapter that implements gguf.FileLike. The adapter
		// currently provides header/tensor index inspection and will be
		// extended to provide full tensor reads.
		// Re-open file for adapter ownership.
		rf, err := os.Open(path)
		if err != nil {
			return nil, fmt.Errorf("open model for legacy parse: %w", err)
		}
		bf, err := parseBin(rf)
		if err != nil {
			rf.Close()
			return nil, err
		}
		return bf, nil
	default:
		return nil, fmt.Errorf("unknown model magic: %q", string(mag[:]))
	}
}

// binFile is a lightweight adapter representing a parsed legacy ggml .bin
// model. It implements gguf.FileLike so the rest of the codebase can query
// metadata and tensor names. Full tensor dequantisation will be added later.
type binFile struct {
	f     *os.File
	meta  map[string]any
	names []string
	tdmap map[string]struct {
		shape  []uint64
		dtype  uint32
		offset uint64
	}
}

func parseBin(f *os.File) (*binFile, error) {
	// Rewind to start and validate magic.
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek start: %w", err)
	}
	var mag [4]byte
	if _, err := io.ReadFull(f, mag[:]); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(mag[:]) != "lmgg" && string(mag[:]) != "GGML" {
		return nil, fmt.Errorf("legacy bin: invalid magic %q", string(mag[:]))
	}

	// Read up to 8 header uint32 fields (some files have fewer). Tolerate
	// short reads — we'll keep the first few values as metadata.
	var header [8]uint32
	for i := 0; i < 8; i++ {
		if err := binary.Read(f, binary.LittleEndian, &header[i]); err != nil {
			break
		}
	}
	meta := make(map[string]any)
	for i := 0; i < 4; i++ {
		meta[fmt.Sprintf("ggml.header.%d", i)] = header[i]
	}

	// Heuristically parse tensor descriptors: the legacy format stores
	// null-terminated names followed by ndim (uint32), dims (uint32[]),
	// dtype (uint32) and offset (uint64). We'll read entries until a
	// sanity check fails.
	tdmap := make(map[string]struct {
		shape  []uint64
		dtype  uint32
		offset uint64
	})
	var names []string

	for {
		// read name using tolerant strategies (NUL-terminated or length-prefixed)
		namePos, _ := f.Seek(0, io.SeekCurrent)
		name, err := readNullString(f, 512)
		if err != nil || name == "" || !looksLikeName(name) {
			// try length-prefixed name (uint32 length + bytes)
			if _, err := f.Seek(namePos, io.SeekStart); err != nil {
				break
			}
			var l uint32
			if err := binary.Read(f, binary.LittleEndian, &l); err != nil {
				break
			}
			if l == 0 || l > 4096 {
				break
			}
			nb := make([]byte, l)
			if _, err := io.ReadFull(f, nb); err != nil {
				break
			}
			name = string(nb)
			// trim any trailing NUL
			if idx := strings.IndexByte(name, 0); idx >= 0 {
				name = name[:idx]
			}
			if name == "" || !looksLikeName(name) {
				break
			}
		}

		var ndim uint32
		if err := binary.Read(f, binary.LittleEndian, &ndim); err != nil {
			break
		}
		if ndim == 0 || ndim > 8 {
			break
		}
		shape := make([]uint64, ndim)
		ok := true
		for i := uint32(0); i < ndim; i++ {
			var d uint32
			if err := binary.Read(f, binary.LittleEndian, &d); err != nil {
				ok = false
				break
			}
			if d == 0 || d > 1_000_000 {
				ok = false
				break
			}
			shape[i] = uint64(d)
		}
		if !ok {
			break
		}
		var dtype uint32
		if err := binary.Read(f, binary.LittleEndian, &dtype); err != nil {
			break
		}
		// accept known dtype codes
		known := dtype == 0 || dtype == 1 || dtype == 2 || dtype == 3 || dtype == 6 || dtype == 7 || dtype == 8 || dtype == 12
		if !known {
			break
		}
		// offset may be stored as uint64 or uint32 in some legacy dumps.
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			// try uint32 fallback
			if _, err2 := f.Seek(-8, io.SeekCurrent); err2 == nil {
				var off32 uint32
				if err3 := binary.Read(f, binary.LittleEndian, &off32); err3 == nil {
					offset = uint64(off32)
				} else {
					break
				}
			} else {
				break
			}
		}
		// If offset looks relative (small and within file), accept it; if it's larger
		// than file size assume it's relative to data start and keep as-is for now.
		fi, _ := f.Stat()
		if offset == 0 {
			break
		}
		if int64(offset) > fi.Size() {
			// allow relative offsets; store as-is
		}

		names = append(names, name)
		tdmap[name] = struct {
			shape  []uint64
			dtype  uint32
			offset uint64
		}{shape: shape, dtype: dtype, offset: offset}
	}

	return &binFile{f: f, meta: meta, names: names, tdmap: tdmap}, nil
}

// Implement gguf.FileLike methods with minimal behaviour.
func (b *binFile) Meta(key string) (any, bool) { v, ok := b.meta[key]; return v, ok }
func (b *binFile) MetaString(key string) (string, bool) {
	v, ok := b.meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}
func (b *binFile) MetaUint32(key string) (uint32, bool) {
	v, ok := b.meta[key]
	if !ok {
		return 0, false
	}
	u, ok := v.(uint32)
	return u, ok
}
func (b *binFile) MetaFloat32(key string) (float32, bool)  { return 0, false }
func (b *binFile) MetaStrings(key string) ([]string, bool) { return nil, false }
func (b *binFile) MetaUint32s(key string) ([]uint32, bool) { return nil, false }
func (b *binFile) TensorNames() []string                   { return b.names }
func (b *binFile) Tensor(ctx context.Context, name string) ([]float32, []int, error) {
	td, ok := b.tdmap[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor %q not found", name)
	}
	num := uint64(1)
	for _, d := range td.shape {
		num *= d
	}
	raw, err := readRawAt(ctx, b.f, int64(td.offset), td.dtype, num)
	if err != nil {
		return nil, nil, err
	}
	out, err := gguf.Dequantize(raw, td.dtype, num)
	if err != nil {
		return nil, nil, err
	}
	shape := make([]int, len(td.shape))
	for i, d := range td.shape {
		shape[i] = int(d)
	}
	return out, shape, nil
}

func (b *binFile) TensorRaw(ctx context.Context, name string) ([]byte, []int, gguf.QuantType, error) {
	td, ok := b.tdmap[name]
	if !ok {
		return nil, nil, gguf.QuantF32, fmt.Errorf("tensor %q not found", name)
	}
	num := uint64(1)
	for _, d := range td.shape {
		num *= d
	}
	raw, err := readRawAt(ctx, b.f, int64(td.offset), td.dtype, num)
	if err != nil {
		return nil, nil, gguf.QuantF32, err
	}

	shape := make([]int, len(td.shape))
	for i, d := range td.shape {
		shape[i] = int(d)
	}
	return raw, shape, gguf.QuantType(td.dtype), nil
}

func (b *binFile) TensorType(name string) (gguf.QuantType, bool) {
	td, ok := b.tdmap[name]
	if !ok {
		return gguf.QuantF32, false
	}
	return gguf.QuantType(td.dtype), true
}

func (b *binFile) Close() error { return b.f.Close() }

// readNullString reads a NUL-terminated string up to maxLen bytes.
func readNullString(r io.Reader, maxLen int) (string, error) {
	buf := make([]byte, 0, maxLen)
	single := make([]byte, 1)
	for i := 0; i < maxLen; i++ {
		if _, err := io.ReadFull(r, single); err != nil {
			return "", err
		}
		if single[0] == 0 {
			break
		}
		buf = append(buf, single[0])
	}
	return string(buf), nil
}

func looksLikeName(s string) bool {
	if len(s) == 0 || len(s) > 200 {
		return false
	}
	for _, r := range s {
		if r == '\u0000' {
			return false
		}
		if r == '/' || r == '\\' {
			return false
		}
		if !unicode.IsPrint(r) {
			return false
		}
	}
	return true
}

// local helpers for raw size and read (copied/adapted from gguf/tensor.go)
func blocksLocal(n uint64) uint64 { return (n + 31) / 32 }
func rawSizeLocal(dtype uint32, n uint64) (uint64, error) {
	switch dtype {
	case 0:
		return n * 4, nil
	case 1:
		return n * 2, nil
	case 2:
		return blocksLocal(n) * 18, nil
	case 3:
		return blocksLocal(n) * 20, nil
	case 6:
		return blocksLocal(n) * 22, nil
	case 7:
		return blocksLocal(n) * 24, nil
	case 8:
		return blocksLocal(n) * 34, nil
	case 12:
		return ((n + 255) / 256) * 148, nil
	default:
		return 0, fmt.Errorf("unsupported dtype: %d", dtype)
	}
}

func readRawAt(ctx context.Context, f *os.File, absOffset int64, dtype uint32, numElems uint64) ([]byte, error) {
	size, err := rawSizeLocal(dtype, numElems)
	if err != nil {
		return nil, err
	}
	if _, err := f.Seek(absOffset, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek: %w", err)
	}
	out := make([]byte, size)
	var read int64
	total := int64(size)
	const chunk = int64(1 << 20)
	for read < total {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled: %w", err)
		}
		end := read + chunk
		if end > total {
			end = total
		}
		if _, err := io.ReadFull(f, out[read:end]); err != nil {
			return nil, fmt.Errorf("read data: %w", err)
		}
		read = end
	}
	return out, nil
}
