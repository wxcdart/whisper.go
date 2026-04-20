//go:build !(darwin && (amd64 || arm64)) && !(linux && (amd64 || arm64))

package ml

// No purego BLAS override is available on this platform.
// sgemmImpl remains set to gonumSgemm (the default in blas.go).
