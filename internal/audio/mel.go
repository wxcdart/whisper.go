package audio

import (
	"context"
	"fmt"
	"math"
	"runtime"

	"golang.org/x/sync/errgroup"
)

// LogMel computes a log-mel spectrogram from mono float32 PCM at SampleRate.
// Uses errgroup to parallelise FFT frame computation across available CPUs.
func LogMel(ctx context.Context, samples []float32, filters MelFilters) (Mel, error) {
	nMel := filters.NMel
	if nMel == 0 {
		nMel = NMel
	}
	nFreq := NFFT/2 + 1 // 201

	// 1. Pad samples to length: n_samples + N_FFT (zero pad right).
	padded := make([]float32, len(samples)+NFFT)
	copy(padded, samples)

	// 2. Compute Hann window.
	hann := hannWindow(NFFT)

	// 3. Determine number of frames.
	nFrames := 1 + (len(padded)-NFFT)/HopLength

	// Allocate output: [nMel × nFrames], row-major.
	data := make([]float32, nMel*nFrames)

	// 4. Parallelise over frame groups.
	nCPU := runtime.NumCPU()
	if nCPU > nFrames {
		nCPU = nFrames
	}

	g, gCtx := errgroup.WithContext(ctx)
	chunkSize := (nFrames + nCPU - 1) / nCPU

	for w := 0; w < nCPU; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > nFrames {
			end = nFrames
		}
		if start >= end {
			break
		}

		tStart, tEnd := start, end // capture for goroutine
		g.Go(func() error {
			// Per-goroutine scratch buffers.
			windowed := make([]float64, NFFT)
			fftSize := nextPow2(NFFT)
			fftBuf := make([]complex128, fftSize)
			power := make([]float64, nFreq)

			for t := tStart; t < tEnd; t++ {
				select {
				case <-gCtx.Done():
					return fmt.Errorf("audio: mel: %w", gCtx.Err())
				default:
				}

				pos := t * HopLength

				// a. Apply Hann window.
				for i := 0; i < NFFT; i++ {
					if pos+i < len(padded) {
						windowed[i] = float64(padded[pos+i]) * hann[i]
					} else {
						windowed[i] = 0
					}
				}

				// b. Zero-pad to fftSize.
				for i := 0; i < fftSize; i++ {
					if i < NFFT {
						fftBuf[i] = complex(windowed[i], 0)
					} else {
						fftBuf[i] = 0
					}
				}

				// c. In-place FFT.
				fftInPlace(fftBuf)

				// d. Power spectrum.
				for k := 0; k < nFreq; k++ {
					re := real(fftBuf[k])
					im := imag(fftBuf[k])
					power[k] = re*re + im*im
				}

				// e. Apply mel filterbank.
				for m := 0; m < nMel; m++ {
					var sum float64
					row := filters.Data[m*nFreq : m*nFreq+nFreq]
					for k := 0; k < nFreq; k++ {
						sum += float64(row[k]) * power[k]
					}
					if sum < 1e-10 {
						sum = 1e-10
					}
					data[m*nFrames+t] = float32(math.Log10(sum))
				}
			}
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return Mel{}, err
	}

	// 5. Normalise.
	maxVal := float32(math.Inf(-1))
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	for i := range data {
		v := data[i]
		if v < maxVal-8.0 {
			v = maxVal - 8.0
		}
		data[i] = (v + 4.0) / 4.0
	}

	return Mel{NMel: nMel, NLen: nFrames, Data: data}, nil
}

// hannWindow returns a Hann window of length n.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// fftInPlace performs an in-place Cooley-Tukey radix-2 DIT FFT.
// len(x) must be a power of 2.
func fftInPlace(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}

	// Bit-reversal permutation.
	bits := 0
	for tmp := n >> 1; tmp > 0; tmp >>= 1 {
		bits++
	}
	for i := 0; i < n; i++ {
		j := bitReverse(i, bits)
		if j > i {
			x[i], x[j] = x[j], x[i]
		}
	}

	// Butterfly stages.
	for s := 2; s <= n; s <<= 1 {
		half := s >> 1
		wBase := complex(math.Cos(-2*math.Pi/float64(s)), math.Sin(-2*math.Pi/float64(s)))
		for k := 0; k < n; k += s {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				t := w * x[k+j+half]
				u := x[k+j]
				x[k+j] = u + t
				x[k+j+half] = u - t
				w *= wBase
			}
		}
	}
}

// bitReverse reverses the bits of v using the given number of bits.
func bitReverse(v, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (v & 1)
		v >>= 1
	}
	return result
}
