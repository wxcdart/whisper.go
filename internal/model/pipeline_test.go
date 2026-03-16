package model

import (
	"context"
	"testing"

	"github.com/whispergo/whisper.go/internal/audio"
	"github.com/whispergo/whisper.go/internal/ml"
	"github.com/whispergo/whisper.go/internal/vocab"
)

// mockEncoder is a minimal encoder for testing.
type mockEncoder struct{}

func (m *mockEncoder) Encode(ctx context.Context, mel ml.Tensor) (ml.Tensor, error) {
	// Return a dummy encoder output [1, 384] (typical shape for tiny model)
	return ml.New(1, 384), nil
}

// mockDecoder is a minimal decoder for testing.
type mockDecoder struct {
	vocab *vocab.Vocabulary
}

func (m *mockDecoder) Decode(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error) {
	// Return a dummy segment
	return []Segment{
		{
			StartMs: 0,
			EndMs:   1000,
			Text:    "hello world",
			Tokens: []TokenData{
				{ID: 50257, Text: " hello", P: 0.9, T0: 0, T1: 500},
				{ID: 50258, Text: " world", P: 0.85, T0: 500, T1: 1000},
			},
		},
	}, nil
}

// TestTranscribe_OutputSegments tests that the pipeline produces segments with timestamps.
func TestTranscribe_OutputSegments(t *testing.T) {
	// Create a minimal vocabulary
	tokens := []string{
		"<|startoftranscript|>",
		"<|en|>",
		"<|transcribe|>",
		"<|notimestamps|>",
		" hello",
		" world",
	}
	tokenTypes := []uint32{1, 1, 1, 1, 0, 0}

	v, err := vocab.New(tokens, tokenTypes)
	if err != nil {
		t.Fatalf("failed to create vocabulary: %v", err)
	}

	// Create pipeline with mock encoder/decoder
	enc := &mockEncoder{}
	dec := &mockDecoder{vocab: v}

	p, err := New(enc, dec, v, nil, nil)
	if err != nil {
		t.Fatalf("failed to create pipeline: %v", err)
	}

	// Create test audio (1 second at 16kHz)
	samples := make([]float32, audio.SampleRate)
	for i := range samples {
		samples[i] = 0.1 * float32(i%1000) / 1000.0 // Small sine-like wave
	}

	// Run transcription
	params := TranscribeParams{
		DecoderParams: DecoderParams{
			BeamSize: 1,
		},
		Language:           "en",
		AutoDetectLanguage: false,
	}

	result, err := p.Transcribe(context.Background(), samples, params)
	if err != nil {
		t.Fatalf("transcription failed: %v", err)
	}

	// Verify output
	if result == nil {
		t.Fatal("result is nil")
	}

	if len(result.Segments) == 0 {
		t.Fatal("no segments produced")
	}

	seg := result.Segments[0]
	if seg.Text == "" {
		t.Error("segment text is empty")
	}

	if seg.StartMs < 0 {
		t.Errorf("negative start time: %d", seg.StartMs)
	}

	if seg.EndMs <= seg.StartMs {
		t.Errorf("invalid time range: [%d, %d]", seg.StartMs, seg.EndMs)
	}

	if len(seg.Tokens) == 0 {
		t.Error("no tokens in segment")
	}
}

// TestTranscribe_ContextCancellation ensures cancellation propagates.
func TestTranscribe_ContextCancellation(t *testing.T) {
	// Create a minimal vocabulary
	tokens := []string{
		"<|startoftranscript|>",
		"<|en|>",
		"<|transcribe|>",
	}
	tokenTypes := []uint32{1, 1, 1}

	v, err := vocab.New(tokens, tokenTypes)
	if err != nil {
		t.Fatalf("failed to create vocabulary: %v", err)
	}

	enc := &mockEncoder{}
	dec := &mockDecoder{vocab: v}

	p, err := New(enc, dec, v, nil, nil)
	if err != nil {
		t.Fatalf("failed to create pipeline: %v", err)
	}

	// Create test audio
	samples := make([]float32, audio.SampleRate*30) // 30 seconds

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Run transcription with cancelled context
	params := TranscribeParams{
		Language: "en",
	}

	_, err = p.Transcribe(ctx, samples, params)
	if err == nil {
		t.Error("expected context cancellation error, got nil")
	}

	if err != context.Canceled {
		t.Errorf("expected context.Canceled, got %v", err)
	}
}
