package ml

// Tensor is a dense, row-major float32 n-dimensional array.
type Tensor struct {
	Data  []float32
	Shape []int // len(Shape) == ndim; product(Shape) == len(Data)
}

// New allocates a zero Tensor with the given shape.
func New(shape ...int) Tensor { panic("not implemented") }

// From wraps an existing slice as a Tensor with the given shape.
// Panics if len(data) != product(shape).
func From(data []float32, shape ...int) Tensor { panic("not implemented") }

// Size returns the total number of elements.
func (t Tensor) Size() int { panic("not implemented") }

// Clone returns a deep copy.
func (t Tensor) Clone() Tensor { panic("not implemented") }

// Reshape returns a view with a new shape. Total element count must be unchanged.
func (t Tensor) Reshape(shape ...int) Tensor { panic("not implemented") }
