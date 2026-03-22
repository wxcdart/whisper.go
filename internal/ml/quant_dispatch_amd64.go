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
		var sel string
		quantQ40DotKernel, sel = chooseAVX2DotKernel(QuantTypeQ4_0, dotQ4_0F32AVX2, dotQ4_0F32Generic)
		quantDebugf("startup kernel_select quant=Q4_0 arch=amd64 selected=%s", sel)
		quantQ41DotKernel, sel = chooseAVX2DotKernel(QuantTypeQ4_1, dotQ4_1F32AVX2, dotQ4_1F32Generic)
		quantDebugf("startup kernel_select quant=Q4_1 arch=amd64 selected=%s", sel)
		quantQ50DotKernel, sel = chooseAVX2DotKernel(QuantTypeQ5_0, dotQ5_0F32AVX2, dotQ5_0F32Generic)
		quantDebugf("startup kernel_select quant=Q5_0 arch=amd64 selected=%s", sel)
		quantQ51DotKernel, sel = chooseAVX2DotKernel(QuantTypeQ5_1, dotQ5_1F32AVX2, dotQ5_1F32Generic)
		quantDebugf("startup kernel_select quant=Q5_1 arch=amd64 selected=%s", sel)
		quantQ80DotKernel, sel = chooseAVX2DotKernel(QuantTypeQ8_0, dotQ8_0F32AVX2, dotQ8_0F32Generic)
		quantDebugf("startup kernel_select quant=Q8_0 arch=amd64 selected=%s", sel)
		quantRuntimeHasSIMD = true
	} else {
		quantDebugf("startup kernel_select arch=amd64 avx2=false selected=generic_all")
	}
}
