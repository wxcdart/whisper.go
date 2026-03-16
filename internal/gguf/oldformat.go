package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
)

// parseOldGGML parses the old ggml binary format and creates a File compatible structure.
// Old ggml format structure:
// [magic: 4 bytes "lmgg"]
// [version: 4 bytes]
// [n_tensors: 4 bytes]
// For each tensor:
//   [n_dims: 4 bytes]
//   [dims: n_dims * 4 bytes]
//   [type: 4 bytes]
//   [name_len: 4 bytes]
//   [name: name_len bytes]
//   [offset: 8 bytes] (to tensor data)
func loadOldGGMLFormat(r io.Reader) ([]tensorDesc, map[string]any, error) {
	tensors := []tensorDesc{}
	metadata := make(map[string]any)

	// Skip magic (already validated in Open)
	if _, err := io.ReadFull(r, make([]byte, 4)); err != nil {
		return nil, nil, fmt.Errorf("skip magic: %w", err)
	}

	// Read version
	version := make([]byte, 4)
	if _, err := io.ReadFull(r, version); err != nil {
		return nil, nil, fmt.Errorf("read version: %w", err)
	}
	metadata["format"] = "old_ggml"
	metadata["version"] = binary.LittleEndian.Uint32(version)

	// Read number of tensors
	nTensorsBytes := make([]byte, 4)
	if _, err := io.ReadFull(r, nTensorsBytes); err != nil {
		return nil, nil, fmt.Errorf("read n_tensors: %w", err)
	}
	numTensors := binary.LittleEndian.Uint32(nTensorsBytes)

	// Parse each tensor
	for i := uint32(0); i < numTensors; i++ {
		// Read number of dimensions
		nDimsBytes := make([]byte, 4)
		if _, err := io.ReadFull(r, nDimsBytes); err != nil {
			return nil, nil, fmt.Errorf("tensor %d: read n_dims: %w", i, err)
		}
		dims := binary.LittleEndian.Uint32(nDimsBytes)

		// Read dimension sizes
		shape := make([]uint64, dims)
		for d := uint32(0); d < dims; d++ {
			dimBuf := make([]byte, 4)
			if _, err := io.ReadFull(r, dimBuf); err != nil {
				return nil, nil, fmt.Errorf("tensor %d: read dim %d: %w", i, d, err)
			}
			shape[d] = uint64(binary.LittleEndian.Uint32(dimBuf))
		}

		// Read tensor type
		typeBuf := make([]byte, 4)
		if _, err := io.ReadFull(r, typeBuf); err != nil {
			return nil, nil, fmt.Errorf("tensor %d: read type: %w", i, err)
		}
		dtype := binary.LittleEndian.Uint32(typeBuf)

		// Read name length
		nameLenBuf := make([]byte, 4)
		if _, err := io.ReadFull(r, nameLenBuf); err != nil {
			return nil, nil, fmt.Errorf("tensor %d: read name_len: %w", i, err)
		}
		nameLen := binary.LittleEndian.Uint32(nameLenBuf)

		// Read tensor name
		nameBuf := make([]byte, nameLen)
		if _, err := io.ReadFull(r, nameBuf); err != nil {
			return nil, nil, fmt.Errorf("tensor %d: read name: %w", i, err)
		}
		name := string(nameBuf)

		// Read data offset
		offsetBuf := make([]byte, 8)
		if _, err := io.ReadFull(r, offsetBuf); err != nil {
			return nil, nil, fmt.Errorf("tensor %d: read offset: %w", i, err)
		}
		offset := binary.LittleEndian.Uint64(offsetBuf)

		// Create tensor descriptor
		desc := tensorDesc{
			name:   name,
			shape:  shape,
			dtype:  oldTypeToTensorType(dtype),
			offset: offset,
		}

		tensors = append(tensors, desc)
	}

	return tensors, metadata, nil
}

// oldTypeToTensorType converts old ggml type codes to GGUF TensorType values.
// These mappings are based on standard ggml type indices.
func oldTypeToTensorType(oldType uint32) uint32 {
	// Map old ggml format type codes to GGUF type constants
	typeMap := map[uint32]uint32{
		0:  0,  // F32
		1:  1,  // F16
		2:  2,  // Q4_0
		3:  3,  // Q4_1
		4:  2,  // Q4_1_NEW (treat as Q4_0)
		5:  5,  // Q5_0
		6:  6,  // Q5_1
		7:  7,  // Q8_0
		8:  8,  // Q8_1
		9:  9,  // Q2_K
		10: 10, // Q3_K
		11: 11, // Q4_K
		12: 12, // Q5_K
		13: 13, // Q6_K
		14: 14, // Q8_K
	}

	if t, ok := typeMap[oldType]; ok {
		return t
	}

	// Default to F32 for unknown types
	return 0
}
