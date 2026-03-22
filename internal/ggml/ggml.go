package ggml

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
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

	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != 0x67676d6c { // "ggml"
		return nil, fmt.Errorf("legacy bin: invalid magic 0x%x", magic)
	}

	// Legacy convert-pt-to-ggml header layout after magic:
	// n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer,
	// n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, use_f16
	var h [11]uint32
	for i := range h {
		if err := binary.Read(f, binary.LittleEndian, &h[i]); err != nil {
			return nil, fmt.Errorf("legacy bin: read hparams[%d]: %w", i, err)
		}
	}

	meta := map[string]any{
		"ggml.magic":                           magic,
		"whisper.vocab.size":                   h[0],
		"whisper.encoder.context_length":       h[1],
		"whisper.encoder.embedding_length":     h[2],
		"whisper.encoder.attention.head_count": h[3],
		"whisper.encoder.layer_count":          h[4],
		"whisper.decoder.context_length":       h[5],
		"whisper.decoder.embedding_length":     h[6],
		"whisper.decoder.attention.head_count": h[7],
		"whisper.decoder.layer_count":          h[8],
		"whisper.audio.mel_count":              h[9],
		"ggml.use_f16":                         h[10],
	}

	// Mel filters block: [rows:int32][cols:int32][rows*cols float32]
	var melRows, melCols uint32
	if err := binary.Read(f, binary.LittleEndian, &melRows); err != nil {
		return nil, fmt.Errorf("legacy bin: read mel rows: %w", err)
	}
	if err := binary.Read(f, binary.LittleEndian, &melCols); err != nil {
		return nil, fmt.Errorf("legacy bin: read mel cols: %w", err)
	}
	meta["ggml.mel.rows"] = melRows
	meta["ggml.mel.cols"] = melCols

	melElems := uint64(melRows) * uint64(melCols)
	melBytes := int64(melElems * 4)
	if melBytes > 0 {
		if _, err := f.Seek(melBytes, io.SeekCurrent); err != nil {
			return nil, fmt.Errorf("legacy bin: skip mel data: %w", err)
		}
	}

	// Tokenizer block: [num_tokens:int32][len:int32][bytes]...
	var numTokens uint32
	if err := binary.Read(f, binary.LittleEndian, &numTokens); err != nil {
		return nil, fmt.Errorf("legacy bin: read tokenizer count: %w", err)
	}
	meta["ggml.token.count"] = numTokens

	for i := uint32(0); i < numTokens; i++ {
		var tokenLen uint32
		if err := binary.Read(f, binary.LittleEndian, &tokenLen); err != nil {
			return nil, fmt.Errorf("legacy bin: read tokenizer len[%d]: %w", i, err)
		}
		if tokenLen > 0 {
			if _, err := f.Seek(int64(tokenLen), io.SeekCurrent); err != nil {
				return nil, fmt.Errorf("legacy bin: skip tokenizer bytes[%d]: %w", i, err)
			}
		}
	}

	tdmap := make(map[string]struct {
		shape  []uint64
		dtype  uint32
		offset uint64
	})
	names := make([]string, 0, 512)
	fileInfo, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("legacy bin: stat file: %w", err)
	}
	fileSize := fileInfo.Size()

	for {
		var nDims, nameLen, ftype uint32
		if err := binary.Read(f, binary.LittleEndian, &nDims); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				break
			}
			return nil, fmt.Errorf("legacy bin: read tensor n_dims: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &nameLen); err != nil {
			return nil, fmt.Errorf("legacy bin: read tensor name_len: %w", err)
		}
		if err := binary.Read(f, binary.LittleEndian, &ftype); err != nil {
			return nil, fmt.Errorf("legacy bin: read tensor ftype: %w", err)
		}

		if nDims == 0 || nDims > 8 || nameLen == 0 || nameLen > 4096 {
			return nil, fmt.Errorf("legacy bin: invalid tensor header n_dims=%d name_len=%d ftype=%d", nDims, nameLen, ftype)
		}

		dimsGGML := make([]uint32, nDims)
		for i := uint32(0); i < nDims; i++ {
			if err := binary.Read(f, binary.LittleEndian, &dimsGGML[i]); err != nil {
				return nil, fmt.Errorf("legacy bin: read tensor dim[%d]: %w", i, err)
			}
			if dimsGGML[i] == 0 {
				return nil, fmt.Errorf("legacy bin: invalid zero tensor dim[%d]", i)
			}
		}

		nameBytes := make([]byte, nameLen)
		if _, err := io.ReadFull(f, nameBytes); err != nil {
			return nil, fmt.Errorf("legacy bin: read tensor name: %w", err)
		}

		name := cleanTensorName(string(nameBytes))
		if name == "" {
			return nil, fmt.Errorf("legacy bin: empty tensor name after cleanup")
		}

		shape := make([]uint64, nDims)
		nElems := uint64(1)
		// Convert from ggml storage order to conventional tensor order.
		for i := uint32(0); i < nDims; i++ {
			d := uint64(dimsGGML[nDims-1-i])
			shape[i] = d
			nElems *= d
		}

		dataOffset, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return nil, fmt.Errorf("legacy bin: get tensor data offset: %w", err)
		}

		rawBytes, err := rawSizeLocal(ftype, nElems)
		if err != nil {
			return nil, fmt.Errorf("legacy bin: tensor %q raw size: %w", name, err)
		}

		// Some legacy writers align tensor data to 32 bytes. Probe both layouts and
		// pick the one that yields a plausible next tensor header.
		alignedOffset := align32(dataOffset)
		if alignedOffset > dataOffset {
			nextNoAlign := dataOffset + int64(rawBytes)
			nextAlign := alignedOffset + int64(rawBytes)
			noAlignOK := probeNextTensorHeader(f, nextNoAlign, fileSize)
			alignOK := probeNextTensorHeader(f, nextAlign, fileSize)
			if alignOK && !noAlignOK {
				dataOffset = alignedOffset
			}
		}

		if _, err := f.Seek(dataOffset+int64(rawBytes), io.SeekStart); err != nil {
			return nil, fmt.Errorf("legacy bin: skip tensor %q data: %w", name, err)
		}

		if _, exists := tdmap[name]; !exists {
			names = append(names, name)
			tdmap[name] = struct {
				shape  []uint64
				dtype  uint32
				offset uint64
			}{
				shape:  shape,
				dtype:  ftype,
				offset: uint64(dataOffset),
			}
		}
	}

	return &binFile{f: f, meta: meta, names: names, tdmap: tdmap}, nil
}

func align32(v int64) int64 {
	const align = int64(32)
	if v%align == 0 {
		return v
	}
	return ((v / align) + 1) * align
}

func isKnownLegacyDType(v uint32) bool {
	switch v {
	case 0, 1, 2, 3, 6, 7, 8, 12:
		return true
	default:
		return false
	}
}

func probeNextTensorHeader(f *os.File, offset, fileSize int64) bool {
	// End-of-file is valid (we're at the final tensor).
	if offset >= fileSize {
		return true
	}
	buf := make([]byte, 12)
	if _, err := f.ReadAt(buf, offset); err != nil {
		return false
	}
	nDims := binary.LittleEndian.Uint32(buf[0:4])
	nameLen := binary.LittleEndian.Uint32(buf[4:8])
	ftype := binary.LittleEndian.Uint32(buf[8:12])
	if nDims == 0 || nDims > 8 {
		return false
	}
	if nameLen == 0 || nameLen > 4096 {
		return false
	}
	if !isKnownLegacyDType(ftype) {
		return false
	}
	return true
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
		// try cleaned variant
		if cleaned := cleanTensorName(name); cleaned != name {
			td, ok = b.tdmap[cleaned]
		}
		if !ok {
			return nil, nil, fmt.Errorf("tensor %q not found", name)
		}
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
		if cleaned := cleanTensorName(name); cleaned != name {
			td, ok = b.tdmap[cleaned]
		}
		if !ok {
			return nil, nil, gguf.QuantF32, fmt.Errorf("tensor %q not found", name)
		}
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
		if cleaned := cleanTensorName(name); cleaned != name {
			td, ok = b.tdmap[cleaned]
		}
		if !ok {
			return gguf.QuantF32, false
		}
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

// cleanTensorName removes non-alphanumeric, non-dot, non-underscore
// characters and collapses repeated dots. Used to normalize mangled
// legacy tensor names (e.g. trailing punctuation introduced by some
// exporters).
func cleanTensorName(s string) string {
	var b strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '.' || r == '_' {
			b.WriteRune(r)
		}
	}
	out := b.String()
	out = strings.Trim(out, ".")
	if out == "" {
		return ""
	}
	// collapse multiple dots
	dotRe := regexp.MustCompile(`\.+`)
	out = dotRe.ReplaceAllString(out, ".")
	return out
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
		// Q4_K (k-quant 4-bit) uses 144 bytes per 256 elements in ggml
		// canonical layout: 2*f16 + 12 scale bytes + 128 bytes of qs.
		return ((n + 255) / 256) * 144, nil
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
