package ml

import "math"

// These arch-specific kernels are structured for vectorization-friendly codegen.
// They currently remain pure Go and serve as dispatch targets.

func softmaxRowFastInPlaceAVX2(row []float32) {
	softmaxRowFastInPlaceWide(row)
}

func softmaxRowExactInPlaceAVX2(row []float32) {
	softmaxRowExactInPlaceWide(row)
}

func softmaxRowFastInPlaceNEON(row []float32) {
	softmaxRowFastInPlaceWide(row)
}

func softmaxRowExactInPlaceNEON(row []float32) {
	softmaxRowExactInPlaceWide(row)
}

func softmaxRowFastInPlaceWide(row []float32) {
	maxVal := rowMaxUnrolled(row)
	n := len(row)
	var sum float32
	i := 0
	for ; i+7 < n; i += 8 {
		x0 := fastExpApprox(row[i] - maxVal)
		x1 := fastExpApprox(row[i+1] - maxVal)
		x2 := fastExpApprox(row[i+2] - maxVal)
		x3 := fastExpApprox(row[i+3] - maxVal)
		x4 := fastExpApprox(row[i+4] - maxVal)
		x5 := fastExpApprox(row[i+5] - maxVal)
		x6 := fastExpApprox(row[i+6] - maxVal)
		x7 := fastExpApprox(row[i+7] - maxVal)
		row[i], row[i+1], row[i+2], row[i+3] = x0, x1, x2, x3
		row[i+4], row[i+5], row[i+6], row[i+7] = x4, x5, x6, x7
		sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
	}
	for ; i < n; i++ {
		x := fastExpApprox(row[i] - maxVal)
		row[i] = x
		sum += x
	}
	if sum == 0 {
		inv := 1.0 / float32(n)
		for i := range row {
			row[i] = inv
		}
		return
	}
	invSum := 1 / sum
	i = 0
	for ; i+7 < n; i += 8 {
		row[i] *= invSum
		row[i+1] *= invSum
		row[i+2] *= invSum
		row[i+3] *= invSum
		row[i+4] *= invSum
		row[i+5] *= invSum
		row[i+6] *= invSum
		row[i+7] *= invSum
	}
	for ; i < n; i++ {
		row[i] *= invSum
	}
}

func softmaxRowExactInPlaceWide(row []float32) {
	maxVal := rowMaxUnrolled(row)
	n := len(row)
	var sum float32
	i := 0
	for ; i+7 < n; i += 8 {
		x0 := float32(math.Exp(float64(row[i] - maxVal)))
		x1 := float32(math.Exp(float64(row[i+1] - maxVal)))
		x2 := float32(math.Exp(float64(row[i+2] - maxVal)))
		x3 := float32(math.Exp(float64(row[i+3] - maxVal)))
		x4 := float32(math.Exp(float64(row[i+4] - maxVal)))
		x5 := float32(math.Exp(float64(row[i+5] - maxVal)))
		x6 := float32(math.Exp(float64(row[i+6] - maxVal)))
		x7 := float32(math.Exp(float64(row[i+7] - maxVal)))
		row[i], row[i+1], row[i+2], row[i+3] = x0, x1, x2, x3
		row[i+4], row[i+5], row[i+6], row[i+7] = x4, x5, x6, x7
		sum += x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7
	}
	for ; i < n; i++ {
		x := float32(math.Exp(float64(row[i] - maxVal)))
		row[i] = x
		sum += x
	}
	if sum == 0 {
		inv := 1.0 / float32(n)
		for i := range row {
			row[i] = inv
		}
		return
	}
	invSum := 1 / sum
	i = 0
	for ; i+7 < n; i += 8 {
		row[i] *= invSum
		row[i+1] *= invSum
		row[i+2] *= invSum
		row[i+3] *= invSum
		row[i+4] *= invSum
		row[i+5] *= invSum
		row[i+6] *= invSum
		row[i+7] *= invSum
	}
	for ; i < n; i++ {
		row[i] *= invSum
	}
}
