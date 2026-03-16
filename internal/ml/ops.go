package ml

import (
	"fmt"
	"math"
)

// Add adds two tensors element-wise. b is broadcast over the leading dimensions of a.
func Add(a, b Tensor) Tensor {
	return elementwise(a, b, func(x, y float32) float32 { return x + y })
}

// Mul multiplies two tensors element-wise. b is broadcast over the leading dimensions of a.
func Mul(a, b Tensor) Tensor {
	return elementwise(a, b, func(x, y float32) float32 { return x * y })
}

// elementwise applies op element-wise, broadcasting b over the leading dims of a when shapes differ.
func elementwise(a, b Tensor, op func(x, y float32) float32) Tensor {
	if equalShapes(a.Shape, b.Shape) {
		out := New(a.Shape...)
		for i := range out.Data {
			out.Data[i] = op(a.Data[i], b.Data[i])
		}
		return out
	}
	bSize := b.Size()
	if a.Size()%bSize != 0 {
		panic(fmt.Sprintf("ml: elementwise: shapes %v and %v incompatible", a.Shape, b.Shape))
	}
	out := New(a.Shape...)
	for i := range out.Data {
		out.Data[i] = op(a.Data[i], b.Data[i%bSize])
	}
	return out
}

// GELU applies the Gaussian Error Linear Unit activation.
// Uses the tanh approximation matching ggml/whisper.cpp:
//
//	x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
func GELU(t Tensor) Tensor {
	const sqrt2OverPi = 0.7978845608028654
	out := New(t.Shape...)
	for i, x := range t.Data {
		inner := sqrt2OverPi * (x + 0.044715*x*x*x)
		out.Data[i] = x * 0.5 * float32(1+math.Tanh(float64(inner)))
	}
	return out
}

// LayerNorm applies layer normalisation over the last dimension.
// t shape [*, C], weight and bias shape [C].
func LayerNorm(t, weight, bias Tensor, eps float32) Tensor {
	C := t.Shape[len(t.Shape)-1]
	rows := t.Size() / C
	out := New(t.Shape...)
	for r := 0; r < rows; r++ {
		row := t.Data[r*C : (r+1)*C]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(C)
		var vari float32
		for _, v := range row {
			d := v - mean
			vari += d * d
		}
		vari /= float32(C)
		std := float32(math.Sqrt(float64(vari + eps)))
		for i, v := range row {
			out.Data[r*C+i] = weight.Data[i]*(v-mean)/std + bias.Data[i]
		}
	}
	return out
}

// Softmax applies numerically stable softmax along the last axis.
func Softmax(t Tensor) Tensor {
	C := t.Shape[len(t.Shape)-1]
	rows := t.Size() / C
	out := New(t.Shape...)
	for r := 0; r < rows; r++ {
		src := t.Data[r*C : (r+1)*C]
		dst := out.Data[r*C : (r+1)*C]
		maxVal := src[0]
		for _, v := range src[1:] {
			if v > maxVal {
				maxVal = v
			}
		}
		var sum float32
		for i, v := range src {
			dst[i] = float32(math.Exp(float64(v - maxVal)))
			sum += dst[i]
		}
		for i := range dst {
			dst[i] /= sum
		}
	}
	return out
}

// Transpose permutes the axes of t and materialises a new contiguous array.
// axes must be a permutation of 0..ndim-1.
func Transpose(t Tensor, axes ...int) Tensor {
	ndim := len(t.Shape)
	if len(axes) != ndim {
		panic(fmt.Sprintf("ml: Transpose: axes length %d != ndim %d", len(axes), ndim))
	}
	seen := make([]bool, ndim)
	for _, ax := range axes {
		if ax < 0 || ax >= ndim {
			panic(fmt.Sprintf("ml: Transpose: axis %d out of range [0, %d)", ax, ndim))
		}
		if seen[ax] {
			panic(fmt.Sprintf("ml: Transpose: duplicate axis %d", ax))
		}
		seen[ax] = true
	}
	newShape := make([]int, ndim)
	for i, ax := range axes {
		newShape[i] = t.Shape[ax]
	}
	// original strides
	strides := make([]int, ndim)
	strides[ndim-1] = 1
	for i := ndim - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * t.Shape[i+1]
	}
	// permuted strides
	pStrides := make([]int, ndim)
	for i, ax := range axes {
		pStrides[i] = strides[ax]
	}
	out := New(newShape...)
	idx := make([]int, ndim)
	for pos := range out.Data {
		inOff := 0
		for i, s := range pStrides {
			inOff += idx[i] * s
		}
		out.Data[pos] = t.Data[inOff]
		for i := ndim - 1; i >= 0; i-- {
			idx[i]++
			if idx[i] < newShape[i] {
				break
			}
			idx[i] = 0
		}
	}
	return out
}

// Concat concatenates tensors along the given axis.
func Concat(axis int, tensors ...Tensor) Tensor {
	if len(tensors) == 0 {
		panic("ml: Concat: no tensors provided")
	}
	ndim := len(tensors[0].Shape)
	if axis < 0 || axis >= ndim {
		panic(fmt.Sprintf("ml: Concat: axis %d out of range [0, %d)", axis, ndim))
	}
	newShape := make([]int, ndim)
	copy(newShape, tensors[0].Shape)
	newShape[axis] = 0
	for _, t := range tensors {
		newShape[axis] += t.Shape[axis]
	}
	out := New(newShape...)
	// inner size: product of dims after axis
	inner := 1
	for i := axis + 1; i < ndim; i++ {
		inner *= newShape[i]
	}
	// outer size: product of dims before axis
	outer := 1
	for i := 0; i < axis; i++ {
		outer *= newShape[i]
	}
	outPos := 0
	for o := 0; o < outer; o++ {
		for _, t := range tensors {
			n := t.Shape[axis] * inner
			inOff := o * n
			copy(out.Data[outPos:outPos+n], t.Data[inOff:inOff+n])
			outPos += n
		}
	}
	return out
}

// equalShapes reports whether two shape slices are identical.
func equalShapes(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
