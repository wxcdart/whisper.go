package ml

import (
	"context"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

const (
	QuantTypeQ4_0 = uint32(2)
	QuantTypeQ4_1 = uint32(3)
	QuantTypeQ5_0 = uint32(6)
	QuantTypeQ5_1 = uint32(7)
	QuantTypeQ8_0 = uint32(8)

	q40BlockElems = 32
	q40BlockBytes = 18
	q41BlockElems = 32
	q41BlockBytes = 20
	q50BlockElems = 32
	q50BlockBytes = 22
	q51BlockElems = 32
	q51BlockBytes = 24
	q80BlockElems = 32
	q80BlockBytes = 34
)

// QuantizedMatrix stores a row-major matrix [Rows, Cols] in GGUF block-quantized format.
// Each row is encoded independently in quantized blocks over Cols.
type QuantizedMatrix struct {
	QuantType uint32
	Rows      int
	Cols      int
	Data      []byte
	rowBytes  int
}

const (
	quantMatMulParallelMinOps = 1 << 14
)

var (
	quantQ40DotKernel = dotQ4_0F32Generic
	quantQ41DotKernel = dotQ4_1F32Generic
	quantQ50DotKernel = dotQ5_0F32Generic
	quantQ51DotKernel = dotQ5_1F32Generic
	quantQ80DotKernel = dotQ8_0F32Generic

	quantRuntimeHasSIMD bool

	quantDecisionOnce sync.Once
	quantDecisionMode int32 // -1 force off, 0 auto, +1 force on

	q40CalibrateOnce sync.Once
	q41CalibrateOnce sync.Once
	q50CalibrateOnce sync.Once
	q51CalibrateOnce sync.Once
	q80CalibrateOnce sync.Once

	q40FasterAtomic atomic.Bool
	q41FasterAtomic atomic.Bool
	q50FasterAtomic atomic.Bool
	q51FasterAtomic atomic.Bool
	q80FasterAtomic atomic.Bool
)

// NewQuantizedMatrix validates metadata and wraps raw quantized bytes.
func NewQuantizedMatrix(quantType uint32, rows, cols int, data []byte) (QuantizedMatrix, error) {
	if rows <= 0 || cols <= 0 {
		return QuantizedMatrix{}, fmt.Errorf("%w: NewQuantizedMatrix: invalid dims rows=%d cols=%d", ErrShapeMismatch, rows, cols)
	}
	rowBytes, err := quantRowBytes(quantType, cols)
	if err != nil {
		return QuantizedMatrix{}, err
	}
	need := rows * rowBytes
	if len(data) < need {
		return QuantizedMatrix{}, fmt.Errorf("%w: NewQuantizedMatrix: short data got=%d need=%d", ErrShapeMismatch, len(data), need)
	}
	return QuantizedMatrix{QuantType: quantType, Rows: rows, Cols: cols, Data: data, rowBytes: rowBytes}, nil
}

// MatMulQuantTransBInto computes out = A @ B^T where:
// - A is [M, K] float32 tensor
// - B is [N, K] quantized row-wise matrix
// - out is [M, N] float32 tensor
func MatMulQuantTransBInto(ctx context.Context, a Tensor, b QuantizedMatrix, out Tensor) error {
	if len(a.Shape) != 2 || len(out.Shape) != 2 {
		return fmt.Errorf("%w: MatMulQuantTransBInto: a=%v out=%v", ErrShapeMismatch, a.Shape, out.Shape)
	}
	m, k := a.Shape[0], a.Shape[1]
	if k != b.Cols {
		return fmt.Errorf("%w: MatMulQuantTransBInto: a.K=%d != b.Cols=%d", ErrShapeMismatch, k, b.Cols)
	}
	if out.Shape[0] != m || out.Shape[1] != b.Rows {
		return fmt.Errorf("%w: MatMulQuantTransBInto: out=%v want=[%d %d]", ErrShapeMismatch, out.Shape, m, b.Rows)
	}

	dotFn, err := dotKernelForType(b.QuantType)
	if err != nil {
		return err
	}

	ops := m * b.Rows
	workers := runtime.GOMAXPROCS(0)
	if workers > m {
		workers = m
	}
	if workers <= 1 || ops < quantMatMulParallelMinOps {
		return matMulQuantTransBSerial(ctx, a, b, out, dotFn)
	}
	return matMulQuantTransBParallel(ctx, a, b, out, dotFn, workers)
}

func matMulQuantTransBSerial(ctx context.Context, a Tensor, b QuantizedMatrix, out Tensor, dotFn func([]byte, []float32) (float32, error)) error {
	m, k := a.Shape[0], a.Shape[1]
	for i := 0; i < m; i++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		aRow := a.Data[i*k : (i+1)*k]
		outRow := out.Data[i*b.Rows : (i+1)*b.Rows]
		for j := 0; j < b.Rows; j++ {
			row := b.Data[j*b.rowBytes : (j+1)*b.rowBytes]
			dot, err := dotFn(row, aRow)
			if err != nil {
				return err
			}
			outRow[j] = dot
		}
	}
	return nil
}

func matMulQuantTransBParallel(ctx context.Context, a Tensor, b QuantizedMatrix, out Tensor, dotFn func([]byte, []float32) (float32, error), workers int) error {
	m, k := a.Shape[0], a.Shape[1]
	var (
		firstErr error
		mu       sync.Mutex
	)
	setErr := func(err error) {
		if err == nil {
			return
		}
		mu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		mu.Unlock()
	}
	getErr := func() error {
		mu.Lock()
		defer mu.Unlock()
		return firstErr
	}

	var wg sync.WaitGroup
	worker := func(start, end int) {
		defer wg.Done()
		for i := start; i < end; i++ {
			if getErr() != nil {
				return
			}
			if err := ctx.Err(); err != nil {
				setErr(err)
				return
			}
			aRow := a.Data[i*k : (i+1)*k]
			outRow := out.Data[i*b.Rows : (i+1)*b.Rows]
			for j := 0; j < b.Rows; j++ {
				row := b.Data[j*b.rowBytes : (j+1)*b.rowBytes]
				dot, err := dotFn(row, aRow)
				if err != nil {
					setErr(err)
					return
				}
				outRow[j] = dot
			}
		}
	}

	wg.Add(workers)
	for w := 0; w < workers; w++ {
		start := (w * m) / workers
		end := ((w + 1) * m) / workers
		go worker(start, end)
	}
	wg.Wait()
	return getErr()
}

func quantRowBytes(quantType uint32, cols int) (int, error) {
	blocks := (cols + 31) / 32
	switch quantType {
	case QuantTypeQ4_0:
		return blocks * q40BlockBytes, nil
	case QuantTypeQ4_1:
		return blocks * q41BlockBytes, nil
	case QuantTypeQ5_0:
		return blocks * q50BlockBytes, nil
	case QuantTypeQ5_1:
		return blocks * q51BlockBytes, nil
	case QuantTypeQ8_0:
		return blocks * q80BlockBytes, nil
	default:
		return 0, fmt.Errorf("%w: unsupported quant type %d", ErrShapeMismatch, quantType)
	}
}

func dotKernelForType(quantType uint32) (func([]byte, []float32) (float32, error), error) {
	switch quantType {
	case QuantTypeQ4_0:
		return quantQ40DotKernel, nil
	case QuantTypeQ4_1:
		return quantQ41DotKernel, nil
	case QuantTypeQ5_0:
		return quantQ50DotKernel, nil
	case QuantTypeQ5_1:
		return quantQ51DotKernel, nil
	case QuantTypeQ8_0:
		return quantQ80DotKernel, nil
	default:
		return nil, fmt.Errorf("%w: unsupported quant type %d", ErrShapeMismatch, quantType)
	}
}

func dotQ4_0F32(q []byte, f []float32) (float32, error) {
	return quantQ40DotKernel(q, f)
}

func dotQ4_0F32Generic(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q40BlockElems - 1) / q40BlockElems
	if len(q) < nBlocks*q40BlockBytes {
		return 0, fmt.Errorf("%w: dotQ4_0F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q40BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		qs := q[bo+2 : bo+18]
		base := b * 32
		rem := n - base
		if rem >= 32 {
			sum += dotQ40Block32(d, qs, f[base:base+32])
			continue
		}
		for i := 0; i < 16; i++ {
			j0 := base + i*2
			if j0 < n {
				sum += d * float32(int32(qs[i]&0xF)-8) * f[j0]
			}
			j1 := j0 + 1
			if j1 < n {
				sum += d * float32(int32(qs[i]>>4)-8) * f[j1]
			}
		}
	}
	return sum, nil
}

func dotQ4_1F32(q []byte, f []float32) (float32, error) {
	return quantQ41DotKernel(q, f)
}

func dotQ4_1F32Generic(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q41BlockElems - 1) / q41BlockElems
	if len(q) < nBlocks*q41BlockBytes {
		return 0, fmt.Errorf("%w: dotQ4_1F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q41BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		m := f16ToF32Local(uint16(q[bo+2]) | uint16(q[bo+3])<<8)
		qs := q[bo+4 : bo+20]
		base := b * 32
		rem := n - base
		if rem >= 32 {
			sum += dotQ41Block32(d, m, qs, f[base:base+32])
			continue
		}
		for i := 0; i < 16; i++ {
			j0 := base + i*2
			if j0 < n {
				sum += (d*float32(qs[i]&0xF) + m) * f[j0]
			}
			j1 := j0 + 1
			if j1 < n {
				sum += (d*float32(qs[i]>>4) + m) * f[j1]
			}
		}
	}
	return sum, nil
}

func dotQ5_0F32(q []byte, f []float32) (float32, error) {
	return quantQ50DotKernel(q, f)
}

func dotQ5_0F32Generic(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q50BlockElems - 1) / q50BlockElems
	if len(q) < nBlocks*q50BlockBytes {
		return 0, fmt.Errorf("%w: dotQ5_0F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q50BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		qh := uint32(q[bo+2]) | uint32(q[bo+3])<<8 | uint32(q[bo+4])<<16 | uint32(q[bo+5])<<24
		qs := q[bo+6 : bo+22]
		base := b * 32
		rem := n - base
		if rem >= 32 {
			sum += dotQ50Block32(d, qh, qs, f[base:base+32])
			continue
		}
		for i := 0; i < 16; i++ {
			hi0 := (qh >> uint(i*2)) & 1
			hi1 := (qh >> uint(i*2+1)) & 1
			j0 := base + i*2
			if j0 < n {
				v0 := int32(uint32(qs[i]&0xF)|(hi0<<4)) - 16
				sum += d * float32(v0) * f[j0]
			}
			j1 := j0 + 1
			if j1 < n {
				v1 := int32(uint32(qs[i]>>4)|(hi1<<4)) - 16
				sum += d * float32(v1) * f[j1]
			}
		}
	}
	return sum, nil
}

func dotQ5_1F32(q []byte, f []float32) (float32, error) {
	return quantQ51DotKernel(q, f)
}

func dotQ5_1F32Generic(q []byte, f []float32) (float32, error) {
	n := len(f)
	if n == 0 {
		return 0, nil
	}
	nBlocks := (n + q51BlockElems - 1) / q51BlockElems
	if len(q) < nBlocks*q51BlockBytes {
		return 0, fmt.Errorf("%w: dotQ5_1F32: short q bytes", ErrShapeMismatch)
	}
	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q51BlockBytes
		d := f16ToF32Local(uint16(q[bo]) | uint16(q[bo+1])<<8)
		m := f16ToF32Local(uint16(q[bo+2]) | uint16(q[bo+3])<<8)
		qh := uint32(q[bo+4]) | uint32(q[bo+5])<<8 | uint32(q[bo+6])<<16 | uint32(q[bo+7])<<24
		qs := q[bo+8 : bo+24]
		base := b * 32
		rem := n - base
		if rem >= 32 {
			sum += dotQ51Block32(d, m, qh, qs, f[base:base+32])
			continue
		}
		for i := 0; i < 16; i++ {
			hi0 := (qh >> uint(i*2)) & 1
			hi1 := (qh >> uint(i*2+1)) & 1
			j0 := base + i*2
			if j0 < n {
				v0 := float32(uint32(qs[i]&0xF) | (hi0 << 4))
				sum += (d*v0 + m) * f[j0]
			}
			j1 := j0 + 1
			if j1 < n {
				v1 := float32(uint32(qs[i]>>4) | (hi1 << 4))
				sum += (d*v1 + m) * f[j1]
			}
		}
	}
	return sum, nil
}

// DotQ8_0F32 computes dot(q8, f32) where q8 is GGUF Q8_0 block-encoded data.
// q8 data layout per 32 elements: f16 scale (little-endian) + 32 int8 values.
func DotQ8_0F32(q8 []byte, f []float32) (float32, error) {
	return quantQ80DotKernel(q8, f)
}

func dotQ8_0F32Generic(q8 []byte, f []float32) (float32, error) {
	if len(f) == 0 {
		return 0, nil
	}
	n := len(f)
	nBlocks := (n + q80BlockElems - 1) / q80BlockElems
	if len(q8) < nBlocks*q80BlockBytes {
		return 0, fmt.Errorf("%w: DotQ8_0F32: q8 bytes short: got=%d need=%d", ErrShapeMismatch, len(q8), nBlocks*q80BlockBytes)
	}

	var sum float32
	for b := 0; b < nBlocks; b++ {
		bo := b * q80BlockBytes
		scale := f16ToF32Local(uint16(q8[bo]) | uint16(q8[bo+1])<<8)
		v := q8[bo+2 : bo+2+q80BlockElems]
		start := b * q80BlockElems
		rem := n - start
		if rem >= q80BlockElems {
			sum += dotQ8Block32(scale, v, f[start:start+q80BlockElems])
			continue
		}
		for i := 0; i < rem; i++ {
			sum += scale * float32(int8(v[i])) * f[start+i]
		}
	}
	return sum, nil
}

// ShouldUseQuantMatMul reports whether quantized matmul should be used for a given shape.
// It applies CPU feature checks and a one-time runtime calibration so quant path is only used when likely faster.
func ShouldUseQuantMatMul(aRows, k, bRows int, quantType uint32) bool {
	mode := quantDecision()
	if mode < 0 {
		return false
	}
	switch quantType {
	case QuantTypeQ4_0, QuantTypeQ4_1, QuantTypeQ5_0, QuantTypeQ5_1, QuantTypeQ8_0:
	default:
		return false
	}
	if aRows <= 0 || k <= 0 || bRows <= 0 {
		return false
	}
	if !quantRuntimeHasSIMD {
		return mode > 0
	}
	if aRows*k*bRows < (1<<16) || k < 256 {
		return mode > 0
	}
	if mode > 0 {
		return true
	}
	switch quantType {
	case QuantTypeQ4_0:
		q40CalibrateOnce.Do(func() {
			q40FasterAtomic.Store(calibrateQuantMatMul(QuantTypeQ4_0))
		})
		return q40FasterAtomic.Load()
	case QuantTypeQ4_1:
		q41CalibrateOnce.Do(func() {
			q41FasterAtomic.Store(calibrateQuantMatMul(QuantTypeQ4_1))
		})
		return q41FasterAtomic.Load()
	case QuantTypeQ5_0:
		q50CalibrateOnce.Do(func() {
			q50FasterAtomic.Store(calibrateQuantMatMul(QuantTypeQ5_0))
		})
		return q50FasterAtomic.Load()
	case QuantTypeQ5_1:
		q51CalibrateOnce.Do(func() {
			q51FasterAtomic.Store(calibrateQuantMatMul(QuantTypeQ5_1))
		})
		return q51FasterAtomic.Load()
	case QuantTypeQ8_0:
		q80CalibrateOnce.Do(func() {
			q80FasterAtomic.Store(calibrateQuantMatMul(QuantTypeQ8_0))
		})
		return q80FasterAtomic.Load()
	default:
		return false
	}
}

func quantDecision() int32 {
	quantDecisionOnce.Do(func() {
		mode := int32(0)
		if v, ok := os.LookupEnv("WHISPERGO_QUANT_MATMUL"); ok {
			if parsed, err := strconv.Atoi(v); err == nil {
				if parsed < 0 {
					mode = -1
				} else if parsed > 0 {
					mode = 1
				}
			}
		}
		atomic.StoreInt32(&quantDecisionMode, mode)
	})
	return atomic.LoadInt32(&quantDecisionMode)
}

func calibrateQuantMatMul(quantType uint32) bool {
	const (
		m = 1
		k = 384
		n = 384
	)
	a := New(m, k)
	b := New(n, k)
	for i := range a.Data {
		a.Data[i] = float32((i%31)-15) * 0.03125
	}
	for i := range b.Data {
		b.Data[i] = float32((i%29)-14) * 0.03125
	}

	rowBytes, err := quantRowBytes(quantType, k)
	if err != nil {
		return false
	}
	bQRaw := make([]byte, n*rowBytes)
	for r := 0; r < n; r++ {
		fillSyntheticQuantRowLocal(bQRaw[r*rowBytes:(r+1)*rowBytes], quantType)
	}
	qmat, err := NewQuantizedMatrix(quantType, n, k, bQRaw)
	if err != nil {
		return false
	}

	ctx := context.Background()
	outQ := New(m, n)
	outF := New(m, n)

	// Warmup
	_ = MatMulQuantTransBInto(ctx, a, qmat, outQ)
	_ = MatMulTransBInto(ctx, a, b, outF)

	const iters = 16
	startQ := time.Now()
	for i := 0; i < iters; i++ {
		if err := MatMulQuantTransBInto(ctx, a, qmat, outQ); err != nil {
			return false
		}
	}
	qDur := time.Since(startQ)

	startF := time.Now()
	for i := 0; i < iters; i++ {
		if err := MatMulTransBInto(ctx, a, b, outF); err != nil {
			return false
		}
	}
	fDur := time.Since(startF)

	return qDur < fDur
}

func fillSyntheticQuantRowLocal(dst []byte, quantType uint32) {
	for i := range dst {
		dst[i] = byte((i * 37) & 0xFF)
	}
	h := float32ToF16Local(1.0)
	switch quantType {
	case QuantTypeQ4_0:
		for off := 0; off+q40BlockBytes <= len(dst); off += q40BlockBytes {
			dst[off] = byte(h)
			dst[off+1] = byte(h >> 8)
		}
	case QuantTypeQ4_1:
		for off := 0; off+q41BlockBytes <= len(dst); off += q41BlockBytes {
			dst[off] = byte(h)
			dst[off+1] = byte(h >> 8)
			dst[off+2] = 0
			dst[off+3] = 0
		}
	case QuantTypeQ5_0:
		for off := 0; off+q50BlockBytes <= len(dst); off += q50BlockBytes {
			dst[off] = byte(h)
			dst[off+1] = byte(h >> 8)
		}
	case QuantTypeQ5_1:
		for off := 0; off+q51BlockBytes <= len(dst); off += q51BlockBytes {
			dst[off] = byte(h)
			dst[off+1] = byte(h >> 8)
			dst[off+2] = 0
			dst[off+3] = 0
		}
	case QuantTypeQ8_0:
		for off := 0; off+q80BlockBytes <= len(dst); off += q80BlockBytes {
			dst[off] = byte(h)
			dst[off+1] = byte(h >> 8)
		}
	}
}

func quantizeQ80RowLocal(dst []byte, row []float32) {
	n := len(row)
	nBlocks := (n + q80BlockElems - 1) / q80BlockElems
	for b := 0; b < nBlocks; b++ {
		start := b * q80BlockElems
		end := start + q80BlockElems
		if end > n {
			end = n
		}
		absmax := float32(0)
		for i := start; i < end; i++ {
			a := float32(math.Abs(float64(row[i])))
			if a > absmax {
				absmax = a
			}
		}
		scale := float32(0)
		if absmax > 0 {
			scale = absmax / 127.0
		}
		h := float32ToF16Local(scale)
		bo := b * q80BlockBytes
		dst[bo] = byte(h)
		dst[bo+1] = byte(h >> 8)
		for i := 0; i < q80BlockElems; i++ {
			idx := start + i
			q := int8(0)
			if idx < n && scale != 0 {
				v := int32(math.Round(float64(row[idx] / scale)))
				if v < -128 {
					v = -128
				}
				if v > 127 {
					v = 127
				}
				q = int8(v)
			}
			dst[bo+2+i] = byte(q)
		}
	}
}

func float32ToF16Local(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits >> 31)
	exp := int(bits>>23) & 0xFF
	mant := bits & 0x7FFFFF
	switch {
	case exp == 255:
		if mant != 0 {
			return sign<<15 | 0x7E00
		}
		return sign<<15 | 0x7C00
	case exp == 0:
		return sign << 15
	}
	exp16 := exp - 127 + 15
	if exp16 >= 31 {
		return sign<<15 | 0x7C00
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign << 15
		}
		shift := uint(1 - exp16)
		mant16 := uint16((mant | 0x800000) >> (13 + shift))
		return sign<<15 | mant16
	}
	mant16 := uint16(mant >> 13)
	if (mant>>12)&1 == 1 {
		mant16++
	}
	if mant16 >= 0x400 {
		mant16 = 0
		exp16++
		if exp16 >= 31 {
			return sign<<15 | 0x7C00
		}
	}
	return sign<<15 | uint16(exp16)<<10 | mant16
}

func dotQ8Block32(scale float32, q []byte, f []float32) float32 {
	var s0, s1, s2, s3 float32
	for i := 0; i < 32; i += 8 {
		s0 += float32(int8(q[i+0])) * f[i+0]
		s1 += float32(int8(q[i+1])) * f[i+1]
		s2 += float32(int8(q[i+2])) * f[i+2]
		s3 += float32(int8(q[i+3])) * f[i+3]
		s0 += float32(int8(q[i+4])) * f[i+4]
		s1 += float32(int8(q[i+5])) * f[i+5]
		s2 += float32(int8(q[i+6])) * f[i+6]
		s3 += float32(int8(q[i+7])) * f[i+7]
	}
	return scale * (s0 + s1 + s2 + s3)
}

func dotQ40Block32(d float32, qs []byte, f []float32) float32 {
	var sum float32
	for i := 0; i < 16; i++ {
		q := qs[i]
		j := i * 2
		sum += float32(int32(q&0xF)-8) * f[j]
		sum += float32(int32(q>>4)-8) * f[j+1]
	}
	return d * sum
}

func dotQ41Block32(d, m float32, qs []byte, f []float32) float32 {
	var sum float32
	for i := 0; i < 16; i++ {
		q := qs[i]
		j := i * 2
		sum += (d*float32(q&0xF) + m) * f[j]
		sum += (d*float32(q>>4) + m) * f[j+1]
	}
	return sum
}

func dotQ50Block32(d float32, qh uint32, qs []byte, f []float32) float32 {
	var sum float32
	for i := 0; i < 16; i++ {
		hi0 := (qh >> uint(i*2)) & 1
		hi1 := (qh >> uint(i*2+1)) & 1
		q := qs[i]
		j := i * 2
		v0 := float32(int32(uint32(q&0xF)|(hi0<<4)) - 16)
		v1 := float32(int32(uint32(q>>4)|(hi1<<4)) - 16)
		sum += d * v0 * f[j]
		sum += d * v1 * f[j+1]
	}
	return sum
}

func dotQ51Block32(d, m float32, qh uint32, qs []byte, f []float32) float32 {
	var sum float32
	for i := 0; i < 16; i++ {
		hi0 := (qh >> uint(i*2)) & 1
		hi1 := (qh >> uint(i*2+1)) & 1
		q := qs[i]
		j := i * 2
		v0 := float32(uint32(q&0xF) | (hi0 << 4))
		v1 := float32(uint32(q>>4) | (hi1 << 4))
		sum += (d*v0 + m) * f[j]
		sum += (d*v1 + m) * f[j+1]
	}
	return sum
}

// MatMulQ8_0TransBInto computes out = A @ B^T where:
// - A is [M, K] float32 tensor
// - B is [N, K] encoded as row-wise Q8_0 bytes in bQ8
// - out is [M, N] float32 tensor
// This is a narrow quantized path for decoder-like projection matrices.
func MatMulQ8_0TransBInto(ctx context.Context, a Tensor, bQ8 []byte, nRowsB, k int, out Tensor) error {
	b, err := NewQuantizedMatrix(QuantTypeQ8_0, nRowsB, k, bQ8)
	if err != nil {
		return err
	}
	return MatMulQuantTransBInto(ctx, a, b, out)
}

func f16ToF32Local(h uint16) float32 {
	s := uint32(h>>15) << 31
	e := uint32((h >> 10) & 0x1F)
	m := uint32(h & 0x3FF)
	switch e {
	case 0:
		if m == 0 {
			return float32FromBits(s)
		}
		k := uint32(1)
		m <<= 1
		for m&0x400 == 0 {
			m <<= 1
			k++
		}
		m &= 0x3FF
		return float32FromBits(s | ((113 - k) << 23) | (m << 13))
	case 31:
		return float32FromBits(s | 0x7F800000 | (m << 13))
	default:
		return float32FromBits(s | ((e + 112) << 23) | (m << 13))
	}
}

func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
