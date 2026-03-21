//go:build amd64

package ml

import "golang.org/x/sys/cpu"

func init() {
	softmaxRowFastKernel = softmaxRowFastInPlaceGeneric
	softmaxRowExactKernel = softmaxRowExactInPlaceGeneric
	if cpu.X86.HasAVX2 {
		softmaxRowFastKernel = softmaxRowFastInPlaceAVX2
		softmaxRowExactKernel = softmaxRowExactInPlaceAVX2
	}
}
