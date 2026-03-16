package audio

import "context"

const (
	SampleRate = 16000
	NFFT       = 400
	HopLength  = 160
	ChunkSize  = 30 // seconds
	NMel       = 80
)

// MelFilters holds the mel filterbank matrix loaded from a model.
type MelFilters struct {
	NMel int
	Data []float32 // [NMel × (NFFT/2+1)]
}

// Mel is a log-mel spectrogram.
type Mel struct {
	NMel int
	NLen int
	Data []float32 // [NMel × NLen], row-major
}

// LoadWAV reads a WAV file, converts to mono float32 PCM, and resamples to SampleRate if needed.
func LoadWAV(ctx context.Context, path string) (samples []float32, sampleRate int, err error) {
	panic("not implemented")
}

// LoadWAVStereo reads a WAV file and returns both channels as float32 (for diarization).
// If mono, both channels are identical.
func LoadWAVStereo(ctx context.Context, path string) (left, right []float32, err error) {
	panic("not implemented")
}

// LogMel computes a log-mel spectrogram from mono float32 PCM at SampleRate.
// Uses errgroup to parallelise FFT frame computation across available CPUs.
func LogMel(ctx context.Context, samples []float32, filters MelFilters) (Mel, error) {
	panic("not implemented")
}
