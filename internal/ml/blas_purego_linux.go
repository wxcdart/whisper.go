//go:build linux && (amd64 || arm64)

package ml

import (
	"github.com/ebitengine/purego"
)

// candidateBLASLibs is the ordered list of CBLAS shared-library names tried on Linux.
// We stop at the first one that opens successfully and exports cblas_sgemm.
var candidateBLASLibs = []string{
	"libopenblas.so.0",
	"libopenblas.so",
	"libblas.so.3",
	"libblas.so",
}

func init() {
	if !blasEnabled() {
		return
	}

	for _, lib := range candidateBLASLibs {
		handle, err := purego.Dlopen(lib, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			continue
		}

		// Verify that cblas_sgemm is actually exported by this library.
		sym, err := purego.Dlsym(handle, "cblas_sgemm")
		if err != nil || sym == 0 {
			_ = purego.Dlclose(handle)
			continue
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
		return // success — stop trying candidates
	}
	// No system BLAS found; gonum fallback remains active.
}
