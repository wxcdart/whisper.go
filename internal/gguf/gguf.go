package gguf

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
)

// Sentinel errors returned by Open.
var (
	ErrInvalidMagic       = errors.New("gguf: invalid magic")
	ErrUnsupportedVersion = errors.New("gguf: unsupported version")
)

const (
	ggufMagic    = "GGUF"
	defaultAlign = uint64(32)
)

// File represents a parsed GGUF model file.
type File struct {
	f           *os.File
	meta        map[string]any
	tensors     []tensorDesc
	tensorIndex map[string]*tensorDesc
	dataStart   int64
}

// Open parses a GGUF file and returns a File ready to read metadata and tensors.
func Open(ctx context.Context, path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open: %w", err)
	}
	file := &File{
		f:           f,
		meta:        make(map[string]any),
		tensorIndex: make(map[string]*tensorDesc),
	}
	if err := file.parse(ctx); err != nil {
		f.Close()
		return nil, err
	}
	return file, nil
}

func (f *File) parse(ctx context.Context) error {
	r := f.f

	var mag [4]byte
	if _, err := io.ReadFull(r, mag[:]); err != nil {
		return fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(mag[:]) != ggufMagic {
		return ErrInvalidMagic
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("gguf: read version: %w", err)
	}
	if version != 2 && version != 3 {
		return fmt.Errorf("%w: %d", ErrUnsupportedVersion, version)
	}

	var tensorCount, metaCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return fmt.Errorf("gguf: read tensor_count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &metaCount); err != nil {
		return fmt.Errorf("gguf: read metadata_count: %w", err)
	}

	for i := uint64(0); i < metaCount; i++ {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("gguf: context cancelled: %w", err)
		}
		key, val, err := readMetaEntry(r)
		if err != nil {
			return fmt.Errorf("gguf: metadata[%d]: %w", i, err)
		}
		f.meta[key] = val
	}

	f.tensors = make([]tensorDesc, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("gguf: context cancelled: %w", err)
		}
		td, err := readTensorDesc(r)
		if err != nil {
			return fmt.Errorf("gguf: tensor[%d]: %w", i, err)
		}
		f.tensors[i] = td
	}
	for i := range f.tensors {
		f.tensorIndex[f.tensors[i].name] = &f.tensors[i]
	}

	pos, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return fmt.Errorf("gguf: seek position: %w", err)
	}
	alignment := defaultAlign
	if v, ok := f.meta["general.alignment"]; ok {
		if u, ok2 := v.(uint32); ok2 && u > 0 {
			alignment = uint64(u)
		}
	}
	aligned := (uint64(pos) + alignment - 1) / alignment * alignment
	f.dataStart = int64(aligned)
	return nil
}

// Meta returns a metadata value by key. Returns (nil, false) if missing.
func (f *File) Meta(key string) (any, bool) {
	v, ok := f.meta[key]
	return v, ok
}

// MetaString returns a metadata string value.
func (f *File) MetaString(key string) (string, bool) {
	v, ok := f.meta[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok
}

// MetaUint32 returns a metadata uint32 value.
func (f *File) MetaUint32(key string) (uint32, bool) {
	v, ok := f.meta[key]
	if !ok {
		return 0, false
	}
	u, ok := v.(uint32)
	return u, ok
}

// MetaFloat32 returns a metadata float32 value.
func (f *File) MetaFloat32(key string) (float32, bool) {
	v, ok := f.meta[key]
	if !ok {
		return 0, false
	}
	fl, ok := v.(float32)
	return fl, ok
}

// MetaStrings returns a metadata string array value.
func (f *File) MetaStrings(key string) ([]string, bool) {
	v, ok := f.meta[key]
	if !ok {
		return nil, false
	}
	ss, ok := v.([]string)
	return ss, ok
}

// MetaUint32s returns a metadata uint32 array value.
func (f *File) MetaUint32s(key string) ([]uint32, bool) {
	v, ok := f.meta[key]
	if !ok {
		return nil, false
	}
	us, ok := v.([]uint32)
	return us, ok
}

// TensorNames returns all tensor names in the file.
func (f *File) TensorNames() []string {
	names := make([]string, len(f.tensors))
	for i, td := range f.tensors {
		names[i] = td.name
	}
	return names
}

// Tensor dequantises and returns float32 data and shape for the named tensor.
func (f *File) Tensor(ctx context.Context, name string) ([]float32, []int, error) {
	td, ok := f.tensorIndex[name]
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	numElems := uint64(1)
	for _, d := range td.shape {
		numElems *= d
	}
	raw, err := readTensorRaw(ctx, f.f, f.dataStart+int64(td.offset), td.dtype, numElems)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
	}
	out, err := dequantize(raw, td.dtype, numElems)
	if err != nil {
		return nil, nil, fmt.Errorf("gguf: tensor %q: %w", name, err)
	}
	shape := make([]int, len(td.shape))
	for i, d := range td.shape {
		shape[i] = int(d)
	}
	return out, shape, nil
}

// Close releases file resources.
func (f *File) Close() error {
	return f.f.Close()
}
