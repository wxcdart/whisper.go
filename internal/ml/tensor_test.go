package ml

import (
	"context"
	"testing"
)

func TestMatMulSmall(t *testing.T) {
	a := New(2, 3)
	b := New(3, 2)

	// A = [[1,2,3],[4,5,6]]
	a.Data[0] = 1
	a.Data[1] = 2
	a.Data[2] = 3
	a.Data[3] = 4
	a.Data[4] = 5
	a.Data[5] = 6

	// B = [[7,8],[9,10],[11,12]]
	b.Data[0] = 7
	b.Data[1] = 8
	b.Data[2] = 9
	b.Data[3] = 10
	b.Data[4] = 11
	b.Data[5] = 12

	c, err := MatMul(context.Background(), a, b)
	if err != nil {
		t.Fatalf("MatMul error: %v", err)
	}

	want := [][]float32{{58, 64}, {139, 154}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if got := c.Data[i*2+j]; got != want[i][j] {
				t.Fatalf("c[%d,%d]=%v want %v", i, j, got, want[i][j])
			}
		}
	}
}
