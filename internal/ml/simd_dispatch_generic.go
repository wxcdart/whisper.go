//go:build !amd64 && !arm64

package ml

func init() {
	softmaxRowFastKernel = softmaxRowFastInPlaceGeneric
	softmaxRowExactKernel = softmaxRowExactInPlaceGeneric
}
