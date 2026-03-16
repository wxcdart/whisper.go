package gguf

// metaType is the GGUF metadata value-type tag used by the writer.
type metaType = uint32

// Writer-side metadata type constants (aliases of the reader-side type* constants).
const (
metaUint8   metaType = typeUint8
metaInt8    metaType = typeInt8
metaUint16  metaType = typeUint16
metaInt16   metaType = typeInt16
metaUint32  metaType = typeUint32
metaInt32   metaType = typeInt32
metaFloat32 metaType = typeFloat32
metaBool    metaType = typeBool
metaString  metaType = typeString
metaArray   metaType = typeArray
metaUint64  metaType = typeUint64
metaInt64   metaType = typeInt64
metaFloat64 metaType = typeFloat64
)

// metaKV is a single metadata entry used when building a writable GGUF file.
type metaKV struct {
key   string
vtype metaType
val   any
}

// arrayVal holds a typed GGUF array value.
type arrayVal struct {
elemType metaType
elems    []any
}

// tensorRecord is a tensor entry used when building a writable GGUF file.
type tensorRecord struct {
name  string
shape []uint64
dtype QuantType
data  []byte
}

// WritableFile is an in-memory GGUF file being constructed for writing.
// Use NewWritableFile, AddWritableMeta, AddWritableTensor, then WriteFile.
type WritableFile struct {
meta    []metaKV
tensors []*tensorRecord
}

// NewWritableFile returns an empty WritableFile.
func NewWritableFile() *WritableFile { return &WritableFile{} }

// AddWritableMeta appends a metadata key-value entry.
func (w *WritableFile) AddWritableMeta(key string, vtype metaType, val any) {
w.meta = append(w.meta, metaKV{key: key, vtype: vtype, val: val})
}

// AddWritableTensor appends a pre-quantised tensor.
func (w *WritableFile) AddWritableTensor(name string, shape []uint64, dtype QuantType, data []byte) {
w.tensors = append(w.tensors, &tensorRecord{name: name, shape: shape, dtype: dtype, data: data})
}

// copyMetaFrom copies all metadata from a parsed File into this WritableFile.
// It preserves the original GGUF type tags by inspecting the Go type of each value.
func (w *WritableFile) copyMetaFrom(f *File) {
for k, v := range f.meta {
w.meta = append(w.meta, metaKV{key: k, vtype: inferMetaType(v), val: v})
}
}

// AddTensor appends a pre-quantised tensor (alias for AddWritableTensor).
func (w *WritableFile) AddTensor(name string, shape []uint64, dtype QuantType, data []byte) {
	w.AddWritableTensor(name, shape, dtype, data)
}
func inferMetaType(v any) metaType {
switch v.(type) {
case uint8:
return metaUint8
case int8:
return metaInt8
case uint16:
return metaUint16
case int16:
return metaInt16
case uint32:
return metaUint32
case int32:
return metaInt32
case float32:
return metaFloat32
case bool:
return metaBool
case string:
return metaString
case uint64:
return metaUint64
case int64:
return metaInt64
case float64:
return metaFloat64
default:
return metaString // fallback
}
}
