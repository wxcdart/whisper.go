package gguf

import "context"

// metaType is the GGUF metadata value type tag.
type metaType uint32

const (
	metaUint8   metaType = 0
	metaInt8    metaType = 1
	metaUint16  metaType = 2
	metaInt16   metaType = 3
	metaUint32  metaType = 4
	metaInt32   metaType = 5
	metaFloat32 metaType = 6
	metaBool    metaType = 7
	metaString  metaType = 8
	metaArray   metaType = 9
	metaUint64  metaType = 10
	metaInt64   metaType = 11
	metaFloat64 metaType = 12
)

// metaKV holds a single GGUF metadata key-value pair.
type metaKV struct {
	key   string
	vtype metaType
	val   any
}

// arrayVal holds a typed GGUF array metadata value.
type arrayVal struct {
	elemType metaType
	elems    []any
}

// tensorRecord holds a single tensor's metadata and raw quantised bytes.
type tensorRecord struct {
	name  string
	shape []uint64
	dtype QuantType
	data  []byte
}

// File represents a parsed GGUF model file.
type File struct {
	meta    []metaKV
	tensors []*tensorRecord
}

// Open parses a GGUF file and returns a File ready to read metadata and tensors.
func Open(ctx context.Context, path string) (*File, error) { panic("not implemented") }

// Meta returns a metadata value by key. Returns (nil, false) if missing.
func (f *File) Meta(key string) (any, bool) { panic("not implemented") }

// MetaString is a convenience helper that returns a metadata string value.
func (f *File) MetaString(key string) (string, bool) { panic("not implemented") }

// MetaUint32 returns a metadata uint32 value.
func (f *File) MetaUint32(key string) (uint32, bool) { panic("not implemented") }

// MetaFloat32 returns a metadata float32 value.
func (f *File) MetaFloat32(key string) (float32, bool) { panic("not implemented") }

// MetaStrings returns a metadata string array value.
func (f *File) MetaStrings(key string) ([]string, bool) { panic("not implemented") }

// MetaUint32s returns a metadata uint32 array value.
func (f *File) MetaUint32s(key string) ([]uint32, bool) { panic("not implemented") }

// Tensor returns the dequantised float32 data and shape for the named tensor.
func (f *File) Tensor(ctx context.Context, name string) (data []float32, shape []int, err error) {
	panic("not implemented")
}

// TensorNames returns all tensor names in the file.
func (f *File) TensorNames() []string { panic("not implemented") }

// TensorType returns the quantisation type of the named tensor.
func (f *File) TensorType(name string) (QuantType, bool) { panic("not implemented") }

// Close releases file resources.
func (f *File) Close() error { panic("not implemented") }

// NewFile returns an empty File suitable for building and writing.
func NewFile() *File { return &File{} }

// AddMeta appends a metadata key-value entry to f.
func (f *File) AddMeta(key string, vtype metaType, val any) {
	f.meta = append(f.meta, metaKV{key: key, vtype: vtype, val: val})
}

// AddTensor appends a tensor to f with pre-quantised raw bytes.
func (f *File) AddTensor(name string, shape []uint64, dtype QuantType, data []byte) {
	f.tensors = append(f.tensors, &tensorRecord{name: name, shape: shape, dtype: dtype, data: data})
}

// copyMetaFrom copies all metadata entries from src into f.
func (f *File) copyMetaFrom(src *File) {
	f.meta = append(f.meta[:0:0], src.meta...)
}
