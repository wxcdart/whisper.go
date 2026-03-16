package model

import (
	"context"

	"github.com/whispergo/whisper.go/internal/ml"
)

// Encoder converts a log-mel spectrogram into encoder hidden states.
type Encoder interface {
	Encode(ctx context.Context, mel ml.Tensor) (out ml.Tensor, err error)
}

// Decoder generates token sequences from encoder output.
type Decoder interface {
	Decode(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error)
}

// DTWAligner performs dynamic time warping alignment.
type DTWAligner interface {
	Align(ctx context.Context, attention [][]float32, logits [][]float32) ([]TokenData, error)
}

// Transcriber runs the full encode+decode pipeline over chunked audio.
type Transcriber interface {
	Transcribe(ctx context.Context, samples []float32, params TranscribeParams) (*Result, error)
}

// Streamer is a real-time transcription interface for incremental audio.
type Streamer interface {
	// Push adds new audio samples to the internal buffer.
	Push(samples []float32) error
	// Results returns a channel for receiving transcribed segments in real-time.
	Results() <-chan Segment
	// Close finalizes the stream and releases resources.
	Close() error
}

// Segment is a transcribed time-stamped span of text.
type Segment struct {
	StartMs int64
	EndMs   int64
	Text    string
	Tokens  []TokenData
	Speaker int
}

// TokenData holds per-token metadata produced by the decoder.
type TokenData struct {
	ID    int32
	Text  string
	P     float32
	PLog  float32
	PT    float32
	PTSum float32
	T0    int64
	T1    int64
	TDTW  int64
}

// Result wraps all segments and detected language.
type Result struct {
	Segments []Segment
	Language string
}

// DecoderParams controls a single decode call.
type DecoderParams struct {
	Prompt         []int32 // SOT + lang + task + [notimestamps]
	BeamSize       int
	BestOf         int
	Temperature    float32
	TemperatureInc float32
	EntropyThold   float32
	LogprobThold   float32
	NoSpeechThold  float32
	NoFallback     bool
	MaxTokens      int
	SuppressNST    bool
	SuppressRegex  string
	DTWEnabled     bool
}

// TranscribeParams is the full set of options for the pipeline.
type TranscribeParams struct {
	DecoderParams
	Language           string
	AutoDetectLanguage bool
	Translate          bool
	MaxLen             int
	SplitOnWord        bool
	NoTimestamps       bool
	InitialPrompt      string
	CarryInitialPrompt bool
	OffsetMs           int
	DurationMs         int
	MaxContext         int
	AudioCtx           int
	Threads            int
	VADEnabled         bool
	VADModelPath       string
}
