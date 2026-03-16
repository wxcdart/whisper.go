package audio

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
