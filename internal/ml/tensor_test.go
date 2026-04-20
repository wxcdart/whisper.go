package ml

import (
	"context"
	"testing"
)

func TestMatMulSmall(t *testing.T) {
	// A = [[1,2,3],[4,5,6]]
	a := From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	// B = [[7,8],[9,10],[11,12]]
	b := From([]float32{7, 8, 9, 10, 11, 12}, 3, 2)

	c, err := MatMul(context.Background(), a, b)
	if err != nil {
		t.Fatalf("MatMul error: %v", err)
	}

	want := []float32{58, 64, 139, 154}
	for i, v := range want {
		if c.Data[i] != v {
			t.Fatalf("c[%d]=%v want %v", i, c.Data[i], v)
		}
	}
}
