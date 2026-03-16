package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// Value type constants matching the GGUF spec.
const (
	typeUint8   = uint32(0)
	typeInt8    = uint32(1)
	typeUint16  = uint32(2)
	typeInt16   = uint32(3)
	typeUint32  = uint32(4)
	typeInt32   = uint32(5)
	typeFloat32 = uint32(6)
	typeBool    = uint32(7)
	typeString  = uint32(8)
	typeArray   = uint32(9)
	typeUint64  = uint32(10)
	typeInt64   = uint32(11)
	typeFloat64 = uint32(12)
)

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("read string length: %w", err)
	}
	if length == 0 {
		return "", nil
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("read string bytes: %w", err)
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype uint32) (any, error) {
	switch vtype {
	case typeUint8:
		var v uint8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt8:
		var v int8
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeUint16:
		var v uint16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt16:
		var v int16
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeUint32:
		var v uint32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt32:
		var v int32
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeFloat32:
		var bits uint32
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, err
		}
		return math.Float32frombits(bits), nil
	case typeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return nil, err
		}
		return v != 0, nil
	case typeString:
		return readString(r)
	case typeArray:
		return readArray(r)
	case typeUint64:
		var v uint64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeInt64:
		var v int64
		return v, binary.Read(r, binary.LittleEndian, &v)
	case typeFloat64:
		var bits uint64
		if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
			return nil, err
		}
		return math.Float64frombits(bits), nil
	default:
		return nil, fmt.Errorf("unknown value type: %d", vtype)
	}
}

func readArray(r io.Reader) (any, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return nil, fmt.Errorf("read array elem type: %w", err)
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read array count: %w", err)
	}
	// Optimised paths for the most common array element types.
	switch elemType {
	case typeString:
		ss := make([]string, count)
		for i := range ss {
			s, err := readString(r)
			if err != nil {
				return nil, fmt.Errorf("array string[%d]: %w", i, err)
			}
			ss[i] = s
		}
		return ss, nil
	case typeUint32:
		us := make([]uint32, count)
		for i := range us {
			if err := binary.Read(r, binary.LittleEndian, &us[i]); err != nil {
				return nil, fmt.Errorf("array uint32[%d]: %w", i, err)
			}
		}
		return us, nil
	case typeFloat32:
		fs := make([]float32, count)
		for i := range fs {
			var bits uint32
			if err := binary.Read(r, binary.LittleEndian, &bits); err != nil {
				return nil, fmt.Errorf("array float32[%d]: %w", i, err)
			}
			fs[i] = math.Float32frombits(bits)
		}
		return fs, nil
	default:
		result := make([]any, count)
		for i := range result {
			v, err := readValue(r, elemType)
			if err != nil {
				return nil, fmt.Errorf("array[%d]: %w", i, err)
			}
			result[i] = v
		}
		return result, nil
	}
}

func readMetaEntry(r io.Reader) (string, any, error) {
	key, err := readString(r)
	if err != nil {
		return "", nil, fmt.Errorf("read key: %w", err)
	}
	var vtype uint32
	if err := binary.Read(r, binary.LittleEndian, &vtype); err != nil {
		return "", nil, fmt.Errorf("read value_type for key %q: %w", key, err)
	}
	val, err := readValue(r, vtype)
	if err != nil {
		return "", nil, fmt.Errorf("key %q: %w", key, err)
	}
	return key, val, nil
}
