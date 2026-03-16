package gguf

import "sort"

// metaType represents the GGUF metadata value type tag.
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

// metaKV is an ordered metadata key-value entry used when writing a GGUF file.
type metaKV struct {
	key   string
	vtype metaType
	val   any
}

// arrayVal holds an array metadata value for writing.
type arrayVal struct {
	elemType metaType
	elems    []any
}

// tensorRecord holds tensor data (raw bytes) ready to be serialised to a GGUF file.
type tensorRecord struct {
	name  string
	shape []uint64
	dtype QuantType
	data  []byte
}

// WritableFile is an in-memory GGUF file representation used for serialisation.
type WritableFile struct {
	meta    []metaKV
	tensors []*tensorRecord
}

// NewFile returns an empty WritableFile.
func NewFile() *WritableFile {
	return &WritableFile{}
}

// copyMetaFrom copies all metadata entries from a parsed read-only File.
// Keys are sorted for deterministic output.
func (w *WritableFile) copyMetaFrom(f *File) {
	keys := make([]string, 0, len(f.meta))
	for k := range f.meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	w.meta = make([]metaKV, 0, len(f.meta))
	for _, k := range keys {
		w.meta = append(w.meta, metaKV{key: k, vtype: inferMetaType(f.meta[k]), val: f.meta[k]})
	}
}

// AddTensor appends a tensor record to the WritableFile.
func (w *WritableFile) AddTensor(name string, shape []uint64, dtype QuantType, data []byte) {
	w.tensors = append(w.tensors, &tensorRecord{name: name, shape: shape, dtype: dtype, data: data})
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
		return metaString
	}
}
