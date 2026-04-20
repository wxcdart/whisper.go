//go:build darwin && (amd64 || arm64)

package ml

import (
	"github.com/ebitengine/purego"
)

// accelerateBLASPath is the canonical Accelerate framework BLAS dylib on macOS.
// Note: This path reflects the layout in macOS 10.x–14.x. If Apple changes the
// framework structure in a future release, the init will fail silently and the
// gonum fallback will remain active.
const accelerateBLASPath = "/System/Library/Frameworks/Accelerate.framework/Versions/A/Libraries/libBLAS.dylib"

func init() {
	if !blasEnabled() {
		return
	}

	handle, err := purego.Dlopen(accelerateBLASPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		// Accelerate not available; keep gonum fallback.
		return
	}

	var cblasSgemm func(
		order, transA, transB int32,
		M, N, K int32,
		alpha float32,
		A *float32, lda int32,
		B *float32, ldb int32,
		beta float32,
		C *float32, ldc int32,
	)
	purego.RegisterLibFunc(&cblasSgemm, handle, "cblas_sgemm")

	sgemmImpl = func(
		transA, transB int32,
		M, N, K int32,
		alpha float32,
		A []float32, lda int32,
		B []float32, ldb int32,
		beta float32,
		C []float32, ldc int32,
	) {
		if len(A) == 0 || len(B) == 0 || len(C) == 0 {
			return
		}
		cblasSgemm(cblasRowMajor, transA, transB, M, N, K, alpha, &A[0], lda, &B[0], ldb, beta, &C[0], ldc)
	}
}
