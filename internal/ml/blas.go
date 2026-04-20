package ml

import (
	"os"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// CBLAS order / transpose constants matching the C enum values defined in
// the CBLAS specification (cblas.h). These correspond to the CblasOrder and
// CblasTranspose enumerations respectively.
const (
	cblasRowMajor int32 = 101 // CblasRowMajor
	cblasNoTrans  int32 = 111 // CblasNoTrans
	cblasTrans    int32 = 112 // CblasTrans
)

// sgemmFunc is the Go-level signature for an SGEMM implementation.
// It maps directly to the cblas_sgemm interface:
//
//	C = alpha*op(A)*op(B) + beta*C
//	A: [M, K], B: [K, N] (or transposed), C: [M, N], all row-major.
type sgemmFunc func(
	transA, transB int32,
	M, N, K int32,
	alpha float32,
	A []float32, lda int32,
	B []float32, ldb int32,
	beta float32,
	C []float32, ldc int32,
)

// sgemmImpl is the active SGEMM implementation.
// It is set to gonumSgemm at package init and may be overridden by a
// platform-specific init (blas_purego_darwin.go, blas_purego_linux.go)
// when a faster system BLAS is available.
var sgemmImpl sgemmFunc = gonumSgemm

// blasEnabled reports whether the purego BLAS path is permitted.
// Set WHISPERGO_BLAS=off to force the gonum fallback.
func blasEnabled() bool {
	return os.Getenv("WHISPERGO_BLAS") != "off"
}

// gonumSgemm is the default SGEMM implementation backed by gonum's pure-Go BLAS.
func gonumSgemm(
	transA, transB int32,
	M, N, K int32,
	alpha float32,
	A []float32, lda int32,
	B []float32, ldb int32,
	beta float32,
	C []float32, ldc int32,
) {
	tA := blas.NoTrans
	if transA == cblasTrans {
		tA = blas.Trans
	}
	tB := blas.NoTrans
	if transB == cblasTrans {
		tB = blas.Trans
	}

	m, n, k := int(M), int(N), int(K)

	var aGen blas32.General
	if tA == blas.NoTrans {
		aGen = blas32.General{Rows: m, Cols: k, Data: A, Stride: int(lda)}
	} else {
		aGen = blas32.General{Rows: k, Cols: m, Data: A, Stride: int(lda)}
	}

	var bGen blas32.General
	if tB == blas.NoTrans {
		bGen = blas32.General{Rows: k, Cols: n, Data: B, Stride: int(ldb)}
	} else {
		bGen = blas32.General{Rows: n, Cols: k, Data: B, Stride: int(ldb)}
	}

	blas32.Gemm(tA, tB, alpha,
		aGen,
		bGen,
		beta,
		blas32.General{Rows: m, Cols: n, Data: C, Stride: int(ldc)},
	)
}

// cblasTranspose converts a transB bool to the matching CBLAS transpose constant.
func cblasTranspose(trans bool) int32 {
	if trans {
		return cblasTrans
	}
	return cblasNoTrans
}
