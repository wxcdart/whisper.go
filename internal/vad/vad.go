package vad

import (
	"context"

	"github.com/whispergo/whisper.go/internal/gguf"
)

// SpeechSegment is a detected speech interval.
type SpeechSegment struct {
	StartMs int64
	EndMs   int64
}

// Params controls VAD behaviour.
type Params struct {
	Threshold      float32
	MinSpeechMs    int
	MinSilenceMs   int
	MaxSpeechS     float32
	SpeechPadMs    int
	SamplesOverlap float32
}

// DefaultParams returns params matching whisper.cpp defaults.
func DefaultParams() Params {
	return Params{
		Threshold:      0.5,
		MinSpeechMs:    250,
		MinSilenceMs:   100,
		MaxSpeechS:     3600,
		SpeechPadMs:    30,
		SamplesOverlap: 0.1,
	}
}

// VAD detects speech segments in audio.
type VAD interface {
	Detect(ctx context.Context, samples []float32, sampleRate int, params Params) ([]SpeechSegment, error)
}

// New loads a Silero VAD model from the given GGUF file.
func New(ctx context.Context, f *gguf.File) (VAD, error) { panic("not implemented") }
