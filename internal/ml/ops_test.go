package ml

import (
	"context"
	"math"
	"testing"
)

// approxEq returns true when |a-b| <= tol.
func approxEq(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// TestTensorBasics covers New, From, Size, Clone, Reshape.
func TestTensorBasics(t *testing.T) {
	t.Run("New", func(t *testing.T) {
		tn := New(2, 3)
		if !equalShapes(tn.Shape, []int{2, 3}) {
			t.Fatalf("shape want [2,3] got %v", tn.Shape)
		}
		if len(tn.Data) != 6 {
			t.Fatalf("data len want 6 got %d", len(tn.Data))
		}
		for _, v := range tn.Data {
			if v != 0 {
				t.Fatal("New should produce zeroed data")
			}
		}
	})

	t.Run("From", func(t *testing.T) {
		data := []float32{1, 2, 3, 4}
		tn := From(data, 2, 2)
		if tn.Data[3] != 4 {
			t.Fatalf("want 4 got %v", tn.Data[3])
		}
		// From shares the slice
		data[0] = 99
		if tn.Data[0] != 99 {
			t.Fatal("From should share the data slice")
		}
	})

	t.Run("FromPanicsOnMismatch", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on shape mismatch")
			}
		}()
		From([]float32{1, 2, 3}, 2, 2)
	})

	t.Run("Size", func(t *testing.T) {
		cases := []struct {
			shape []int
			want  int
		}{
			{[]int{3}, 3},
			{[]int{2, 4}, 8},
			{[]int{2, 3, 4}, 24},
		}
		for _, c := range cases {
			tn := New(c.shape...)
			if got := tn.Size(); got != c.want {
				t.Errorf("shape %v Size()=%d want %d", c.shape, got, c.want)
			}
		}
	})

	t.Run("Clone", func(t *testing.T) {
		orig := From([]float32{1, 2, 3, 4}, 2, 2)
		cl := orig.Clone()
		if !equalShapes(cl.Shape, orig.Shape) {
			t.Fatal("clone shape mismatch")
		}
		// data independence
		orig.Data[0] = 99
		if cl.Data[0] == 99 {
			t.Fatal("Clone must deep-copy data")
		}
	})

	t.Run("Reshape", func(t *testing.T) {
		orig := New(2, 6)
		r := orig.Reshape(3, 4)
		if !equalShapes(r.Shape, []int{3, 4}) {
			t.Fatalf("shape want [3,4] got %v", r.Shape)
		}
		// same underlying data
		r.Data[0] = 7
		if orig.Data[0] != 7 {
			t.Fatal("Reshape must share data slice")
		}
	})

	t.Run("ReshapePanicsOnSizeMismatch", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic on size mismatch")
			}
		}()
		New(2, 3).Reshape(5)
	})
}

// TestMatMul covers 2-D, identity, and batched cases.
func TestMatMul(t *testing.T) {
	ctx := context.Background()

	t.Run("2x3_by_3x4", func(t *testing.T) {
		// A=[2,3], B=[3,4]  C=A@B
		a := From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
		b := From([]float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
		}, 3, 4)
		// expected rows:
		// [1+10+27, 2+12+30, 3+14+33, 4+16+36] = [38, 44, 50, 56]
		// [4+25+54, 8+30+60, 12+35+66, 16+40+72] = [83, 98, 113, 128]
		want := []float32{38, 44, 50, 56, 83, 98, 113, 128}
		got, err := MatMul(ctx, a, b)
		if err != nil {
			t.Fatal(err)
		}
		if !equalShapes(got.Shape, []int{2, 4}) {
			t.Fatalf("shape want [2,4] got %v", got.Shape)
		}
		for i, w := range want {
			if !approxEq(got.Data[i], w, 1e-3) {
				t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
			}
		}
	})

	t.Run("3x3_identity", func(t *testing.T) {
		a := From([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
		eye := From([]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}, 3, 3)
		got, err := MatMul(ctx, a, eye)
		if err != nil {
			t.Fatal(err)
		}
		for i, w := range a.Data {
			if !approxEq(got.Data[i], w, 1e-5) {
				t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
			}
		}
	})

	t.Run("batched_2x2x3_by_2x3x2", func(t *testing.T) {
		// batch 0: A[0]=[2,3], B[0]=[3,2]
		// A[0]=[[1,0,0],[0,1,0]], B[0]=[[1,0],[0,1],[0,0]] => I2
		// batch 1: A[1]=[[1,2,3],[4,5,6]], B[1]=[[1,0],[0,1],[1,0]]
		// C[1,0]=[1+0+3, 0+2+0]=[4,2], C[1,1]=[4+0+6, 0+5+0]=[10,5]
		aData := []float32{
			1, 0, 0, 0, 1, 0, // batch 0
			1, 2, 3, 4, 5, 6, // batch 1
		}
		bData := []float32{
			1, 0, 0, 1, 0, 0, // batch 0: [[1,0],[0,1],[0,0]]
			1, 0, 0, 1, 1, 0, // batch 1: [[1,0],[0,1],[1,0]]
		}
		a := From(aData, 2, 2, 3)
		b := From(bData, 2, 3, 2)
		want := []float32{
			1, 0, 0, 1, // batch 0: identity
			4, 2, 10, 5, // batch 1
		}
		got, err := MatMul(ctx, a, b)
		if err != nil {
			t.Fatal(err)
		}
		if !equalShapes(got.Shape, []int{2, 2, 2}) {
			t.Fatalf("shape want [2,2,2] got %v", got.Shape)
		}
		for i, w := range want {
			if !approxEq(got.Data[i], w, 1e-3) {
				t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
			}
		}
	})
}

// TestAdd covers same-shape and broadcast cases.
func TestAdd(t *testing.T) {
	t.Run("same_shape", func(t *testing.T) {
		a := From([]float32{1, 2, 3, 4}, 2, 2)
		b := From([]float32{10, 20, 30, 40}, 2, 2)
		got := Add(a, b)
		want := []float32{11, 22, 33, 44}
		for i, w := range want {
			if got.Data[i] != w {
				t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
			}
		}
	})

	t.Run("broadcast_C_over_BTC", func(t *testing.T) {
		// a: [2,3,4], b: [4]
		a := New(2, 3, 4)
		for i := range a.Data {
			a.Data[i] = 1
		}
		b := From([]float32{1, 2, 3, 4}, 4)
		got := Add(a, b)
		if !equalShapes(got.Shape, []int{2, 3, 4}) {
			t.Fatalf("shape want [2,3,4] got %v", got.Shape)
		}
		want := []float32{2, 3, 4, 5}
		for i := range got.Data {
			if !approxEq(got.Data[i], want[i%4], 1e-6) {
				t.Errorf("[%d] want %v got %v", i, want[i%4], got.Data[i])
			}
		}
	})
}

// TestGELU checks known reference values with tolerance 1e-3.
func TestGELU(t *testing.T) {
	cases := []struct {
		in   float32
		want float32
	}{
		{0, 0},
		{1, 0.8413},
		{-1, -0.1587},
		{2, 1.9545},
	}
	for _, c := range cases {
		tn := From([]float32{c.in}, 1)
		got := GELU(tn)
		if !approxEq(got.Data[0], c.want, 1e-3) {
			t.Errorf("GELU(%v) = %v, want %v", c.in, got.Data[0], c.want)
		}
	}
}

// TestLayerNorm verifies mean≈0, std≈1 for weight=1 bias=0, and offset with weight/bias.
func TestLayerNorm(t *testing.T) {
	const eps = 1e-5

	t.Run("identity_norm", func(t *testing.T) {
		x := From([]float32{1, 2, 3, 4}, 1, 4)
		w := From([]float32{1, 1, 1, 1}, 4)
		bi := From([]float32{0, 0, 0, 0}, 4)
		got := LayerNorm(x, w, bi, eps)
		// verify mean ≈ 0 and variance ≈ 1
		var mean, vari float32
		for _, v := range got.Data {
			mean += v
		}
		mean /= 4
		for _, v := range got.Data {
			d := v - mean
			vari += d * d
		}
		vari /= 4
		if !approxEq(mean, 0, 1e-5) {
			t.Errorf("mean want 0 got %v", mean)
		}
		if !approxEq(vari, 1, 1e-4) {
			t.Errorf("var want 1 got %v", vari)
		}
	})

	t.Run("with_weight_and_bias", func(t *testing.T) {
		// input [1,4] = [1,2,3,4]
		// weight=2, bias=1 → out = 2*normalised + 1
		// normalised ≈ [-1.34164, -0.44721, 0.44721, 1.34164]
		// out ≈ [-1.68328, 0.10557, 1.89442, 3.68328]
		x := From([]float32{1, 2, 3, 4}, 1, 4)
		w := From([]float32{2, 2, 2, 2}, 4)
		bi := From([]float32{1, 1, 1, 1}, 4)
		got := LayerNorm(x, w, bi, eps)
		want := []float32{-1.68328, 0.10557, 1.89442, 3.68328}
		for i, wv := range want {
			if !approxEq(got.Data[i], wv, 1e-3) {
				t.Errorf("[%d] want %v got %v", i, wv, got.Data[i])
			}
		}
	})
}

// TestSoftmax checks known probabilities and that probabilities sum to 1.
func TestSoftmax(t *testing.T) {
	t.Run("known_values", func(t *testing.T) {
		// input [1,0,0] → [e/(e+2), 1/(e+2), 1/(e+2)]
		// numerically ≈ [0.5762, 0.2119, 0.2119]
		x := From([]float32{1, 0, 0}, 1, 3)
		got := Softmax(x)
		want := []float32{0.5762, 0.2119, 0.2119}
		for i, w := range want {
			if !approxEq(got.Data[i], w, 2e-3) {
				t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
			}
		}
	})

	t.Run("sum_to_one", func(t *testing.T) {
		x := From([]float32{3, 1, 0.2}, 1, 3)
		got := Softmax(x)
		var sum float32
		for _, v := range got.Data {
			sum += v
		}
		if !approxEq(sum, 1, 1e-5) {
			t.Errorf("softmax sum want 1 got %v", sum)
		}
	})
}

// TestConv1D verifies a simple [1,0,-1] edge-detector kernel.
func TestConv1D(t *testing.T) {
	ctx := context.Background()

	// input: 1-channel, T=5
	input := From([]float32{1, 2, 3, 4, 5}, 1, 5)
	// weight: [outC=1, inC=1, kernel=3] = [[[ 1, 0, -1 ]]]
	weight := From([]float32{1, 0, -1}, 1, 1, 3)
	bias := From([]float32{0}, 1)

	got, err := Conv1D(ctx, input, weight, bias, 1)
	if err != nil {
		t.Fatal(err)
	}
	// pad=1, T'=5
	// output[0,t] = input[t-1]*1 + input[t]*0 + input[t+1]*(-1)  (with zero padding)
	// t=0: 0 + 0 -2 = -2
	// t=1: 1 + 0 -3 = -2
	// t=2: 2 + 0 -4 = -2
	// t=3: 3 + 0 -5 = -2
	// t=4: 4 + 0 - 0 = 4
	want := []float32{-2, -2, -2, -2, 4}
	if !equalShapes(got.Shape, []int{1, 5}) {
		t.Fatalf("shape want [1,5] got %v", got.Shape)
	}
	for i, w := range want {
		if !approxEq(got.Data[i], w, 1e-5) {
			t.Errorf("[%d] want %v got %v", i, w, got.Data[i])
		}
	}
}

// TestAttention verifies output shape and that the causal mask zeroes the upper triangle of weights.
func TestAttention(t *testing.T) {
	ctx := context.Background()
	// 1 head, T_q=2, T_k=2, head_dim=4
	// Q = K = V: head 0, row 0 = [1,0,0,0], row 1 = [0,1,0,0]
	qData := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	q := From(qData, 1, 2, 4)
	k := From(append([]float32(nil), qData...), 1, 2, 4)
	v := From(append([]float32(nil), qData...), 1, 2, 4)

	t.Run("output_shape", func(t *testing.T) {
		out, _, err := ScaledDotProductAttention(ctx, q, k, v, false, false)
		if err != nil {
			t.Fatal(err)
		}
		if !equalShapes(out.Shape, []int{1, 2, 4}) {
			t.Fatalf("out shape want [1,2,4] got %v", out.Shape)
		}
	})

	t.Run("causal_mask_zeroes_upper_triangle", func(t *testing.T) {
		out, weights, err := ScaledDotProductAttention(ctx, q, k, v, true, true)
		if err != nil {
			t.Fatal(err)
		}
		if !equalShapes(out.Shape, []int{1, 2, 4}) {
			t.Fatalf("out shape want [1,2,4] got %v", out.Shape)
		}
		if !equalShapes(weights.Shape, []int{1, 2, 2}) {
			t.Fatalf("weights shape want [1,2,2] got %v", weights.Shape)
		}
		// weights[head=0, i=0, j=1] must be ~0 (causal: j>i is masked)
		w01 := weights.Data[0*2*2+0*2+1]
		if !approxEq(w01, 0, 1e-5) {
			t.Errorf("causal weights[0,0,1] want 0 got %v", w01)
		}
		// weights[0,0,0] must be ~1 (only position)
		w00 := weights.Data[0*2*2+0*2+0]
		if !approxEq(w00, 1, 1e-5) {
			t.Errorf("causal weights[0,0,0] want 1 got %v", w00)
		}
	})

	t.Run("weights_sum_to_one", func(t *testing.T) {
		_, weights, err := ScaledDotProductAttention(ctx, q, k, v, false, true)
		if err != nil {
			t.Fatal(err)
		}
		// each row of weights (over Tk) must sum to 1
		Tq, Tk := 2, 2
		for h := 0; h < 1; h++ {
			for i := 0; i < Tq; i++ {
				var sum float32
				for j := 0; j < Tk; j++ {
					sum += weights.Data[h*Tq*Tk+i*Tk+j]
				}
				if !approxEq(sum, 1, 1e-5) {
					t.Errorf("head %d row %d: weights sum %v != 1", h, i, sum)
				}
			}
		}
	})

	t.Run("output_values", func(t *testing.T) {
		// With causal=true:
		// row i=0: weights=[1,0], out[0]=V[0,0]=[1,0,0,0]
		// row i=1: out[1] = w10*V[0,0] + w11*V[0,1]
		//   scores (pre-softmax): [0*scale, 0.5*scale*... ] → depends on scale=0.5
		//   scale=1/sqrt(4)=0.5
		//   scores[1,0]=dot([0,1,0,0],[1,0,0,0])*0.5=0, scores[1,1]=dot([0,1,0,0],[0,1,0,0])*0.5=0.5
		//   softmax([0,0.5]) = [1/(1+e^0.5), e^0.5/(1+e^0.5)]
		e05 := float32(math.Exp(0.5))
		w10 := float32(1) / (1 + e05)
		w11 := e05 / (1 + e05)
		out, _, err := ScaledDotProductAttention(ctx, q, k, v, true, false)
		if err != nil {
			t.Fatal(err)
		}
		// out[0,0] = [1,0,0,0]
		if !approxEq(out.Data[0], 1, 1e-5) {
			t.Errorf("out[0,0,0] want 1 got %v", out.Data[0])
		}
		// out[0,1] = w10*[1,0,0,0] + w11*[0,1,0,0] = [w10, w11, 0, 0]
		if !approxEq(out.Data[4], w10, 1e-4) {
			t.Errorf("out[0,1,0] want %v got %v", w10, out.Data[4])
		}
		if !approxEq(out.Data[5], w11, 1e-4) {
			t.Errorf("out[0,1,1] want %v got %v", w11, out.Data[5])
		}
	})
}
