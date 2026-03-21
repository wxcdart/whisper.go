package ml

import (
	"fmt"
	"math"
)

// AddInto computes dst = a + b for equal-shaped tensors.
func AddInto(dst, a, b Tensor) error {
	if !equalShapes(a.Shape, b.Shape) || !equalShapes(dst.Shape, a.Shape) {
		return fmt.Errorf("%w: AddInto: dst=%v a=%v b=%v", ErrShapeMismatch, dst.Shape, a.Shape, b.Shape)
	}
	for i := range dst.Data {
		dst.Data[i] = a.Data[i] + b.Data[i]
	}
	return nil
}

// AddInPlace performs dst += src for equal-shaped tensors.
func AddInPlace(dst, src Tensor) error {
	if !equalShapes(dst.Shape, src.Shape) {
		return fmt.Errorf("%w: AddInPlace: dst=%v src=%v", ErrShapeMismatch, dst.Shape, src.Shape)
	}
	for i := range dst.Data {
		dst.Data[i] += src.Data[i]
	}
	return nil
}

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

// GELUInto applies GELU to src and writes into dst.
func GELUInto(dst, src Tensor) error {
	if !equalShapes(dst.Shape, src.Shape) {
		return fmt.Errorf("%w: GELUInto: dst=%v src=%v", ErrShapeMismatch, dst.Shape, src.Shape)
	}
	const sqrt2OverPi = 0.7978845608028654
	n := len(src.Data)
	i := 0
	for ; i+3 < n; i += 4 {
		x0, x1, x2, x3 := src.Data[i], src.Data[i+1], src.Data[i+2], src.Data[i+3]
		i0 := sqrt2OverPi * (x0 + 0.044715*x0*x0*x0)
		i1 := sqrt2OverPi * (x1 + 0.044715*x1*x1*x1)
		i2 := sqrt2OverPi * (x2 + 0.044715*x2*x2*x2)
		i3 := sqrt2OverPi * (x3 + 0.044715*x3*x3*x3)
		dst.Data[i] = x0 * 0.5 * float32(1+math.Tanh(float64(i0)))
		dst.Data[i+1] = x1 * 0.5 * float32(1+math.Tanh(float64(i1)))
		dst.Data[i+2] = x2 * 0.5 * float32(1+math.Tanh(float64(i2)))
		dst.Data[i+3] = x3 * 0.5 * float32(1+math.Tanh(float64(i3)))
	}
	for ; i < n; i++ {
		x := src.Data[i]
		inner := sqrt2OverPi * (x + 0.044715*x*x*x)
		dst.Data[i] = x * 0.5 * float32(1+math.Tanh(float64(inner)))
	}
	return nil
}

// GELUInPlace applies GELU in place.
func GELUInPlace(t Tensor) {
	const sqrt2OverPi = 0.7978845608028654
	n := len(t.Data)
	i := 0
	for ; i+3 < n; i += 4 {
		x0, x1, x2, x3 := t.Data[i], t.Data[i+1], t.Data[i+2], t.Data[i+3]
		i0 := sqrt2OverPi * (x0 + 0.044715*x0*x0*x0)
		i1 := sqrt2OverPi * (x1 + 0.044715*x1*x1*x1)
		i2 := sqrt2OverPi * (x2 + 0.044715*x2*x2*x2)
		i3 := sqrt2OverPi * (x3 + 0.044715*x3*x3*x3)
		t.Data[i] = x0 * 0.5 * float32(1+math.Tanh(float64(i0)))
		t.Data[i+1] = x1 * 0.5 * float32(1+math.Tanh(float64(i1)))
		t.Data[i+2] = x2 * 0.5 * float32(1+math.Tanh(float64(i2)))
		t.Data[i+3] = x3 * 0.5 * float32(1+math.Tanh(float64(i3)))
	}
	for ; i < n; i++ {
		x := t.Data[i]
		inner := sqrt2OverPi * (x + 0.044715*x*x*x)
		t.Data[i] = x * 0.5 * float32(1+math.Tanh(float64(inner)))
	}
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

// LayerNormInto applies layer normalization over the last dimension into dst.
// t and dst must have the same shape. weight and bias must have shape [C].
func LayerNormInto(dst, t, weight, bias Tensor, eps float32) error {
	if !equalShapes(dst.Shape, t.Shape) {
		return fmt.Errorf("%w: LayerNormInto: dst=%v t=%v", ErrShapeMismatch, dst.Shape, t.Shape)
	}
	if len(t.Shape) == 0 {
		return fmt.Errorf("%w: LayerNormInto: t must have at least 1 dimension", ErrShapeMismatch)
	}
	C := t.Shape[len(t.Shape)-1]
	if len(weight.Shape) != 1 || len(bias.Shape) != 1 || weight.Shape[0] != C || bias.Shape[0] != C {
		return fmt.Errorf("%w: LayerNormInto: weight=%v bias=%v expected [%d]", ErrShapeMismatch, weight.Shape, bias.Shape, C)
	}

	rows := t.Size() / C
	for r := 0; r < rows; r++ {
		start := r * C
		srcRow := t.Data[start : start+C : start+C]
		dstRow := dst.Data[start : start+C : start+C]
		var mean float32
		i := 0
		for ; i+3 < C; i += 4 {
			mean += srcRow[i] + srcRow[i+1] + srcRow[i+2] + srcRow[i+3]
		}
		for ; i < C; i++ {
			mean += srcRow[i]
		}
		mean /= float32(C)

		var vari float32
		i = 0
		for ; i+3 < C; i += 4 {
			d0 := srcRow[i] - mean
			d1 := srcRow[i+1] - mean
			d2 := srcRow[i+2] - mean
			d3 := srcRow[i+3] - mean
			vari += d0*d0 + d1*d1 + d2*d2 + d3*d3
		}
		for ; i < C; i++ {
			d := srcRow[i] - mean
			vari += d * d
		}
		vari /= float32(C)

		std := float32(math.Sqrt(float64(vari + eps)))
		for i, v := range srcRow {
			dstRow[i] = weight.Data[i]*(v-mean)/std + bias.Data[i]
		}
	}

	return nil
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
