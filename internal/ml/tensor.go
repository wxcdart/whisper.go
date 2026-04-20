package ml

import "fmt"

// ErrShapeMismatch is returned when tensor shapes are incompatible for an operation.
var ErrShapeMismatch = fmt.Errorf("ml: shape mismatch")

// Tensor is a dense, row-major float32 n-dimensional array.
type Tensor struct {
	Data  []float32
	Shape []int // len(Shape) == ndim; product(Shape) == len(Data)
}

// New allocates a zero Tensor with the given shape.
func New(shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return Tensor{Data: make([]float32, n), Shape: s}
}

// From wraps an existing slice as a Tensor with the given shape.
// Panics if len(data) != product(shape).
func From(data []float32, shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(data) != n {
		panic(fmt.Sprintf("ml: From: data length %d != product of shape %v (%d)", len(data), shape, n))
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return Tensor{Data: data, Shape: s}
}

// Size returns the total number of elements.
func (t Tensor) Size() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// Clone returns a deep copy.
func (t Tensor) Clone() Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	s := make([]int, len(t.Shape))
	copy(s, t.Shape)
	return Tensor{Data: d, Shape: s}
}

// Reshape returns a zero-copy view with a new shape. Total element count must be unchanged.
func (t Tensor) Reshape(shape ...int) Tensor {
	n := 1
	for _, d := range shape {
		n *= d
	}
	if n != t.Size() {
		panic(fmt.Sprintf("ml: Reshape: new shape %v (size %d) != original size %d", shape, n, t.Size()))
	}
	s := make([]int, len(shape))
	copy(s, shape)
	return Tensor{Data: t.Data, Shape: s}
}
