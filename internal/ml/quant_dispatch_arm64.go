//go:build arm64

package ml

import "golang.org/x/sys/cpu"

func init() {
	quantQ40DotKernel = dotQ4_0F32Generic
	quantQ41DotKernel = dotQ4_1F32Generic
	quantQ50DotKernel = dotQ5_0F32Generic
	quantQ51DotKernel = dotQ5_1F32Generic
	quantQ80DotKernel = dotQ8_0F32Generic
	quantRuntimeHasSIMD = false
	if cpu.ARM64.HasASIMD {
		quantQ40DotKernel = dotQ4_0F32NEON
		quantDebugf("startup kernel_select quant=Q4_0 arch=arm64 selected=neon")
		quantQ41DotKernel = dotQ4_1F32NEON
		quantDebugf("startup kernel_select quant=Q4_1 arch=arm64 selected=neon")
		quantQ50DotKernel = dotQ5_0F32NEON
		quantDebugf("startup kernel_select quant=Q5_0 arch=arm64 selected=neon")
		quantQ51DotKernel = dotQ5_1F32NEON
		quantDebugf("startup kernel_select quant=Q5_1 arch=arm64 selected=neon")
		quantQ80DotKernel = dotQ8_0F32NEON
		quantDebugf("startup kernel_select quant=Q8_0 arch=arm64 selected=neon")
		quantRuntimeHasSIMD = true
	} else {
		quantDebugf("startup kernel_select arch=arm64 asimd=false selected=generic_all")
	}
}
