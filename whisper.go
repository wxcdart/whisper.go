package whisper

import "context"

// SampleRate is the required input audio sample rate.
const SampleRate = 16000

// Context holds a loaded model and is safe for concurrent use across multiple Transcribe calls.
type Context struct{ /* unexported */ }

// Params controls transcription behaviour.
type Params struct {
	Language        string // BCP-47 language code, empty = auto-detect
	Task            Task
	Threads         int
	Processors      int
	BeamSize        int
	BestOf          int
	Temperature     float32
	TemperatureInc  float32
	EntropyThold    float32
	LogprobThold    float32
	NoSpeechThold   float32
	NoFallback      bool
	MaxLen          int
	SplitOnWord     bool
	NoTimestamps    bool
	InitialPrompt   string
	CarryInitialPrompt bool
	OffsetMs        int
	DurationMs      int
	MaxContext      int
	AudioCtx        int
	SuppressNST     bool
	SuppressRegex   string
	DTWPreset       string
	// VAD
	VADEnabled        bool
	VADModelPath      string
	VADThreshold      float32
	VADMinSpeechMs    int
	VADMinSilenceMs   int
	VADMaxSpeechS     float32
	VADSpeechPadMs    int
	VADSamplesOverlap float32
	// Grammar
	Grammar        string
	GrammarRule    string
	GrammarPenalty float32
	// Callbacks
	NewSegment func(ctx context.Context, result *Result, segIdx int) error
	Progress   func(ctx context.Context, progress int) error
}

// Task is the transcription task type.
type Task int

const (
	TaskTranscribe Task = iota
	TaskTranslate
)

// DefaultParams returns sensible defaults matching whisper.cpp defaults.
func DefaultParams() Params {
	return Params{
		Language:          "en",
		Task:              TaskTranscribe,
		Threads:           4,
		BeamSize:          5,
		BestOf:            5,
		Temperature:       0.0,
		TemperatureInc:    0.2,
		EntropyThold:      2.40,
		LogprobThold:      -1.00,
		NoSpeechThold:     0.6,
		VADThreshold:      0.5,
		VADMinSpeechMs:    250,
		VADMinSilenceMs:   100,
		VADSpeechPadMs:    30,
		VADSamplesOverlap: 0.1,
	}
}

// Result is the full transcription output.
type Result struct {
	Segments []Segment
	Language string
}

// Segment is one transcribed segment with timestamps.
type Segment struct {
	StartMs int64
	EndMs   int64
	Text    string
	Tokens  []TokenData
	Speaker int // -1 if unknown; used by diarize
}

// TokenData holds per-token metadata.
type TokenData struct {
	ID    int32
	Text  string
	P     float32 // probability
	PLog  float32 // log probability
	PT    float32 // timestamp probability
	PTSum float32
	T0    int64 // start ms
	T1    int64 // end ms
	TDTW  int64 // DTW-aligned time ms
}

// New loads a model from the given GGUF file path.
func New(ctx context.Context, modelPath string, params ...Params) (*Context, error) {
	panic("not implemented")
}

// Transcribe transcribes audio samples (mono float32 at 16 kHz) and returns the result.
func (c *Context) Transcribe(ctx context.Context, samples []float32, params Params) (*Result, error) {
	panic("not implemented")
}

// Close releases all resources held by the context.
func (c *Context) Close() error { panic("not implemented") }
