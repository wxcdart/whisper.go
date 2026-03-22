//go:build amd64

package ml

import "golang.org/x/sys/cpu"

func init() {
	quantQ40DotKernel = dotQ4_0F32Generic
	quantQ41DotKernel = dotQ4_1F32Generic
	quantQ50DotKernel = dotQ5_0F32Generic
	quantQ51DotKernel = dotQ5_1F32Generic
	quantQ80DotKernel = dotQ8_0F32Generic
	quantRuntimeHasSIMD = false
	if cpu.X86.HasAVX2 {
		quantQ40DotKernel = dotQ4_0F32AVX2
		quantQ41DotKernel = dotQ4_1F32AVX2
		quantQ50DotKernel = dotQ5_0F32AVX2
		quantQ51DotKernel = dotQ5_1F32AVX2
		quantQ80DotKernel = dotQ8_0F32AVX2
		quantRuntimeHasSIMD = true
	}
}
