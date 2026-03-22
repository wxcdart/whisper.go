package mlproto

import "fmt"

// MatMul computes C = A * B using a simple triple-loop algorithm.
// A: m x n, B: n x p -> C: m x p
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Cols() != b.Rows() {
		return nil, fmt.Errorf("matmul: incompatible dims %dx%d * %dx%d", a.Rows(), a.Cols(), b.Rows(), b.Cols())
	}
	m := a.Rows()
	n := a.Cols()
	p := b.Cols()
	c := NewTensor(m, p)

	for i := 0; i < m; i++ {
		for k := 0; k < n; k++ {
			aik := a.At(i, k)
			baseC := i * p
			for j := 0; j < p; j++ {
				c.Data[baseC+j] += aik * b.At(k, j)
			}
		}
	}

	return c, nil
}
