package gguf

import "context"

// File represents a parsed GGUF model file.
type File struct{ /* unexported */ }

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

// Close releases file resources.
func (f *File) Close() error { panic("not implemented") }
