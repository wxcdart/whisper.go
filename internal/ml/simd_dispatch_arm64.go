//go:build arm64

package ml

import "golang.org/x/sys/cpu"

func init() {
	softmaxRowFastKernel = softmaxRowFastInPlaceGeneric
	softmaxRowExactKernel = softmaxRowExactInPlaceGeneric
	if cpu.ARM64.HasASIMD {
		softmaxRowFastKernel = softmaxRowFastInPlaceNEON
		softmaxRowExactKernel = softmaxRowExactInPlaceNEON
	}
}
