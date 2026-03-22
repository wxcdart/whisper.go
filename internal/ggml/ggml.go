package ggml

import (
    "context"
    "encoding/binary"
    "errors"
    "fmt"
    "io"
    "os"

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
    f      *os.File
    meta   map[string]any
    names  []string
    // token: placeholder for future tensor descriptors
}

func parseBin(f *os.File) (*binFile, error) {
    // We already read 4 bytes of magic in Open; rewind to start.
    if _, err := f.Seek(0, io.SeekStart); err != nil {
        return nil, fmt.Errorf("seek start: %w", err)
    }

    // Minimal header parsing: read magic and a few header fields so we can
    // expose some metadata. We avoid full tensor parsing for now.
    var mag [4]byte
    if _, err := io.ReadFull(f, mag[:]); err != nil {
        return nil, fmt.Errorf("read magic: %w", err)
    }
    if string(mag[:]) != "lmgg" && string(mag[:]) != "GGML" {
        return nil, fmt.Errorf("legacy bin: invalid magic %q", string(mag[:]))
    }

    // Read a few 32-bit fields commonly present in ggml headers to populate
    // metadata. This is intentionally tolerant — exact field meanings will be
    // resolved while implementing full parser.
    var header [8]uint32
    if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
        return nil, fmt.Errorf("read header: %w", err)
    }

    meta := make(map[string]any)
    meta["ggml.header.0"] = header[0]
    meta["ggml.header.1"] = header[1]
    meta["ggml.header.2"] = header[2]
    meta["ggml.header.3"] = header[3]

    return &binFile{f: f, meta: meta}, nil
}

// Implement gguf.FileLike methods with minimal behaviour.
func (b *binFile) Meta(key string) (any, bool) { v, ok := b.meta[key]; return v, ok }
func (b *binFile) MetaString(key string) (string, bool) {
    v, ok := b.meta[key]
    if !ok { return "", false }
    s, ok := v.(string)
    return s, ok
}
func (b *binFile) MetaUint32(key string) (uint32, bool) {
    v, ok := b.meta[key]
    if !ok { return 0, false }
    u, ok := v.(uint32)
    return u, ok
}
func (b *binFile) MetaFloat32(key string) (float32, bool) { return 0, false }
func (b *binFile) MetaStrings(key string) ([]string, bool) { return nil, false }
func (b *binFile) MetaUint32s(key string) ([]uint32, bool) { return nil, false }
func (b *binFile) TensorNames() []string { return b.names }
func (b *binFile) Tensor(ctx context.Context, name string) ([]float32, []int, error) { return nil, nil, fmt.Errorf("legacy ggml: tensor read not implemented") }
func (b *binFile) TensorRaw(ctx context.Context, name string) ([]byte, []int, gguf.QuantType, error) { return nil, nil, gguf.QuantF32, fmt.Errorf("legacy ggml: tensor raw read not implemented") }
func (b *binFile) TensorType(name string) (gguf.QuantType, bool) { return gguf.QuantF32, false }
func (b *binFile) Close() error { return b.f.Close() }
