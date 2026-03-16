package audio

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"testing"
)

// writeWAV writes a minimal PCM WAV file and returns its path.
func writeWAV(t *testing.T, samples []int16, channels int, sampleRate int) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "test*.wav")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	nSamples := len(samples)
	dataSize := uint32(nSamples * 2)
	totalSize := 36 + dataSize

	write := func(v any) {
		if werr := binary.Write(f, binary.LittleEndian, v); werr != nil {
			t.Fatal(werr)
		}
	}

	f.Write([]byte("RIFF"))
	write(totalSize)
	f.Write([]byte("WAVE"))
	f.Write([]byte("fmt "))
	write(uint32(16))          // chunk size
	write(uint16(1))           // PCM
	write(uint16(channels))    // num channels
	write(uint32(sampleRate))  // sample rate
	write(uint32(uint32(sampleRate) * uint32(channels) * 2)) // byte rate
	write(uint16(uint16(channels) * 2)) // block align
	write(uint16(16))          // bits per sample
	f.Write([]byte("data"))
	write(dataSize)
	for _, s := range samples {
		write(s)
	}

	return f.Name()
}

// writeStereoWAV writes a stereo 16-bit PCM WAV file.
func writeStereoWAV(t *testing.T, left, right []int16, sampleRate int) string {
	t.Helper()
	interleaved := make([]int16, len(left)*2)
	for i := range left {
		interleaved[i*2] = left[i]
		interleaved[i*2+1] = right[i]
	}
	return writeWAV(t, interleaved, 2, sampleRate)
}

func TestLoadWAV(t *testing.T) {
	t.Run("round-trip 440Hz sine 16-bit PCM", func(t *testing.T) {
		const freq = 440.0
		const sr = 16000
		const dur = 1.0
		n := int(sr * dur)
		raw := make([]int16, n)
		for i := range raw {
			raw[i] = int16(math.Round(0.5 * 32767 * math.Sin(2*math.Pi*freq*float64(i)/float64(sr))))
		}
		path := writeWAV(t, raw, 1, sr)

		samples, rate, err := LoadWAV(context.Background(), path)
		if err != nil {
			t.Fatalf("LoadWAV: %v", err)
		}
		if rate != SampleRate {
			t.Fatalf("rate = %d, want %d", rate, SampleRate)
		}
		if len(samples) != n {
			t.Fatalf("len(samples) = %d, want %d", len(samples), n)
		}
		const tol = 1.0 / 32768.0
		for i, s := range samples {
			want := float32(raw[i]) / 32768.0
			if diff := s - want; diff < -tol || diff > tol {
				t.Fatalf("sample[%d] = %f, want %f (diff %f)", i, s, want, diff)
			}
		}
	})

	t.Run("stereo downmix", func(t *testing.T) {
		n := 100
		left := make([]int16, n)
		right := make([]int16, n)
		for i := range left {
			left[i] = int16(i * 100)
			right[i] = int16(i * 200)
		}
		path := writeStereoWAV(t, left, right, SampleRate)

		samples, _, err := LoadWAV(context.Background(), path)
		if err != nil {
			t.Fatalf("LoadWAV: %v", err)
		}
		const tol = 1.0 / 32768.0
		for i := range samples {
			want := (float32(left[i]) + float32(right[i])) / 2.0 / 32768.0
			if diff := samples[i] - want; diff < -tol || diff > tol {
				t.Fatalf("sample[%d] = %f, want %f", i, samples[i], want)
			}
		}
	})

	t.Run("resample 22050->16000", func(t *testing.T) {
		const srcRate = 22050
		const dur = 1.0
		n := int(srcRate * dur)
		raw := make([]int16, n)
		for i := range raw {
			raw[i] = int16(math.Round(32767 * math.Sin(2*math.Pi*440*float64(i)/float64(srcRate))))
		}
		path := writeWAV(t, raw, 1, srcRate)

		samples, rate, err := LoadWAV(context.Background(), path)
		if err != nil {
			t.Fatalf("LoadWAV: %v", err)
		}
		if rate != SampleRate {
			t.Fatalf("rate = %d, want %d", rate, SampleRate)
		}
		wantLen := int(math.Round(float64(n) * float64(SampleRate) / float64(srcRate)))
		if len(samples) != wantLen {
			t.Fatalf("len(samples) = %d, want ~%d", len(samples), wantLen)
		}
	})

	t.Run("stereo channels", func(t *testing.T) {
		n := 100
		left := make([]int16, n)
		right := make([]int16, n)
		for i := range left {
			left[i] = int16(i * 100)
			right[i] = int16(i * 50)
		}
		path := writeStereoWAV(t, left, right, SampleRate)

		l, r, err := LoadWAVStereo(context.Background(), path)
		if err != nil {
			t.Fatalf("LoadWAVStereo: %v", err)
		}
		const tol = 1.0 / 32768.0
		for i := range l {
			wantL := float32(left[i]) / 32768.0
			wantR := float32(right[i]) / 32768.0
			if diff := l[i] - wantL; diff < -tol || diff > tol {
				t.Fatalf("left[%d] = %f, want %f", i, l[i], wantL)
			}
			if diff := r[i] - wantR; diff < -tol || diff > tol {
				t.Fatalf("right[%d] = %f, want %f", i, r[i], wantR)
			}
		}
	})
}

func TestLogMel(t *testing.T) {
	// Build a synthetic identity-like filterbank for shape testing.
	nFreq := NFFT/2 + 1 // 201

	makeFilters := func(nMel int) MelFilters {
		data := make([]float32, nMel*nFreq)
		for m := 0; m < nMel; m++ {
			k := m * nFreq / nMel
			if k < nFreq {
				data[m*nFreq+k] = 1.0
			}
		}
		return MelFilters{NMel: nMel, Data: data}
	}

	t.Run("mel shape 30s", func(t *testing.T) {
		n := SampleRate * ChunkSize
		samples := make([]float32, n)
		// populate with a sine wave
		for i := range samples {
			samples[i] = float32(math.Sin(2 * math.Pi * 440 * float64(i) / float64(SampleRate)))
		}
		filters := makeFilters(NMel)

		mel, err := LogMel(context.Background(), samples, filters)
		if err != nil {
			t.Fatalf("LogMel: %v", err)
		}
		if mel.NMel != NMel {
			t.Fatalf("NMel = %d, want %d", mel.NMel, NMel)
		}
		// nFrames = 1 + (len(padded) - NFFT) / HopLength
		padLen := n + NFFT
		wantFrames := 1 + (padLen-NFFT)/HopLength
		if mel.NLen != wantFrames {
			t.Fatalf("NLen = %d, want %d", mel.NLen, wantFrames)
		}
		if len(mel.Data) != NMel*wantFrames {
			t.Fatalf("len(Data) = %d, want %d", len(mel.Data), NMel*wantFrames)
		}
	})

	t.Run("mel values in range", func(t *testing.T) {
		const sr = SampleRate
		n := sr // 1 second
		samples := make([]float32, n)
		for i := range samples {
			samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*float64(i)/float64(sr)))
		}
		filters := makeFilters(NMel)

		mel, err := LogMel(context.Background(), samples, filters)
		if err != nil {
			t.Fatalf("LogMel: %v", err)
		}
		for i, v := range mel.Data {
			if v < -2.0 || v > 2.0 {
				t.Fatalf("mel.Data[%d] = %f out of expected range [-2, 2]", i, v)
			}
		}
	})
}
