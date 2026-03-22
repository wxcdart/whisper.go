package ml

import "testing"

func TestMatMulSmall(t *testing.T) {
    a := NewTensor(2, 3)
    b := NewTensor(3, 2)

    // A = [[1,2,3],[4,5,6]]
    a.Set(0, 0, 1)
    a.Set(0, 1, 2)
    a.Set(0, 2, 3)
    a.Set(1, 0, 4)
    a.Set(1, 1, 5)
    a.Set(1, 2, 6)

    // B = [[7,8],[9,10],[11,12]]
    b.Set(0, 0, 7)
    b.Set(0, 1, 8)
    b.Set(1, 0, 9)
    b.Set(1, 1, 10)
    b.Set(2, 0, 11)
    b.Set(2, 1, 12)

    c, err := MatMul(a, b)
    if err != nil {
        t.Fatalf("MatMul error: %v", err)
    }

    want := [][]float32{{58, 64}, {139, 154}}
    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            if got := c.At(i, j); got != want[i][j] {
                t.Fatalf("c[%d,%d]=%v want %v", i, j, got, want[i][j])
            }
        }
    }
}
