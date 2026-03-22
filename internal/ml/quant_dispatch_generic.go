//go:build !amd64 && !arm64

package ml

func init() {
	quantQ40DotKernel = dotQ4_0F32Generic
	quantQ41DotKernel = dotQ4_1F32Generic
	quantQ50DotKernel = dotQ5_0F32Generic
	quantQ51DotKernel = dotQ5_1F32Generic
	quantQ80DotKernel = dotQ8_0F32Generic
	quantRuntimeHasSIMD = false
}
