package mlproto

import "fmt"

// Tensor is a simple 2D row-major float32 tensor used for prototyping.
type Tensor struct {
    Data  []float32
    Shape []int // [rows, cols]
}

// NewTensor allocates a Tensor with given rows and cols.
func NewTensor(rows, cols int) *Tensor {
    return &Tensor{Data: make([]float32, rows*cols), Shape: []int{rows, cols}}
}

func (t *Tensor) Rows() int { return t.Shape[0] }
func (t *Tensor) Cols() int { return t.Shape[1] }

func (t *Tensor) At(r, c int) float32 {
    return t.Data[r*t.Cols()+c]
}

func (t *Tensor) Set(r, c int, v float32) {
    t.Data[r*t.Cols()+c] = v
}

func (t *Tensor) String() string { return fmt.Sprintf("Tensor(shape=%v)", t.Shape) }
