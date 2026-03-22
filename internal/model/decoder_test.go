package model

import (
	"context"
	"errors"
	"testing"

	"github.com/whispergo/whisper.go/internal/ml"
	"github.com/whispergo/whisper.go/internal/vocab"
)

const (
	testNVocab      = 51865 // Realistic Whisper vocab size
	testNTextState  = 4
	testNTextHead   = 2
	testNTextLayer  = 1
	testNTextCtx    = 128
	testTextHeadDim = testNTextState / testNTextHead // 2
)

// buildTestDecoder constructs a WhisperDecoder with tiny dimensions and nonzero weights.
func buildTestDecoder() *WhisperDecoder {
	small := func(shape ...int) ml.Tensor {
		t := ml.New(shape...)
		for i := range t.Data {
			t.Data[i] = 0.01
		}
		return t
	}
	ones := func(n int) ml.Tensor {
		t := ml.New(n)
		for i := range t.Data {
			t.Data[i] = 1.0
		}
		return t
	}
	zeros := func(shape ...int) ml.Tensor { return ml.New(shape...) }

	block := decoderBlock{
		// Self-attention
		sAttnLnW: ones(testNTextState),
		sAttnLnB: zeros(testNTextState),
		selfAttn: decoderSelfAttn{
			qW:   small(testNTextState, testNTextState),
			qB:   zeros(testNTextState),
			kW:   small(testNTextState, testNTextState),
			vW:   small(testNTextState, testNTextState),
			vB:   zeros(testNTextState),
			outW: small(testNTextState, testNTextState),
			outB: zeros(testNTextState),
		},
		// Cross-attention
		cAttnLnW: ones(testNTextState),
		cAttnLnB: zeros(testNTextState),
		crossAttn: decoderCrossAttn{
			qW:   small(testNTextState, testNTextState),
			qB:   zeros(testNTextState),
			kW:   small(testNTextState, testNTextState),
			vW:   small(testNTextState, testNTextState),
			vB:   zeros(testNTextState),
			outW: small(testNTextState, testNTextState),
			outB: zeros(testNTextState),
		},
		// MLP
		mlpLnW: ones(testNTextState),
		mlpLnB: zeros(testNTextState),
		mlp0W:  small(4*testNTextState, testNTextState),
		mlp0B:  zeros(4 * testNTextState),
		mlp2W:  small(testNTextState, 4*testNTextState),
		mlp2B:  zeros(testNTextState),
	}

	return &WhisperDecoder{
		nVocab:     testNVocab,
		nTextState: testNTextState,
		nHead:      testNTextHead,
		nTextLayer: testNTextLayer,
		nTextCtx:   testNTextCtx,
		tokenEmb:   small(testNVocab, testNTextState),
		posEmb:     small(testNTextCtx, testNTextState),
		blocks:     []decoderBlock{block},
		lnW:        ones(testNTextState),
		lnB:        zeros(testNTextState),
		eotToken:   50256,
	}
}

// TestDecode_Greedy_OutputSegments verifies that greedy decoding produces non-empty segments.
func TestDecode_Greedy_OutputSegments(t *testing.T) {
	decoder := buildTestDecoder()

	// Create mock encoder output [T, nTextState]
	encLen := 8
	encoderOut := ml.New(encLen, testNTextState)
	for i := range encoderOut.Data {
		encoderOut.Data[i] = 0.01
	}

	// Set up minimal decoder parameters with a prompt.
	params := DecoderParams{
		Prompt:      []int32{50258, 50259, 50357, 50363}, // Example Whisper prompt
		BeamSize:    1,                                   // Greedy
		MaxTokens:   5,
		Temperature: 1.0,
	}

	ctx := context.Background()
	segments, err := decoder.Decode(ctx, encoderOut, params)
	if err != nil {
		t.Fatalf("Decode returned unexpected error: %v", err)
	}

	// Check that we got at least one segment.
	if len(segments) == 0 {
		t.Error("expected at least one segment, got none")
	}

	// Check that the segment has tokens (should have generated tokens beyond prompt).
	if len(segments) > 0 && len(segments[0].Tokens) == 0 {
		t.Error("expected segment to contain tokens")
	}
}

// TestDecode_ContextCancellation verifies that context cancellation is respected.
func TestDecode_ContextCancellation(t *testing.T) {
	decoder := buildTestDecoder()

	// Create mock encoder output.
	encLen := 8
	encoderOut := ml.New(encLen, testNTextState)
	for i := range encoderOut.Data {
		encoderOut.Data[i] = 0.01
	}

	params := DecoderParams{
		Prompt:    []int32{50258, 50259, 50357, 50363},
		BeamSize:  1,
		MaxTokens: 10,
	}

	// Create and cancel context immediately.
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel before Decode

	_, err := decoder.Decode(ctx, encoderOut, params)
	if err == nil {
		t.Fatal("Decode should have returned an error for cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled in error chain; got: %v", err)
	}
}

// TestDecode_BeamSearch verifies beam search produces non-empty results.
func TestDecode_BeamSearch(t *testing.T) {
	decoder := buildTestDecoder()

	encLen := 8
	encoderOut := ml.New(encLen, testNTextState)
	for i := range encoderOut.Data {
		encoderOut.Data[i] = 0.01
	}

	params := DecoderParams{
		Prompt:      []int32{50258, 50259, 50357, 50363},
		BeamSize:    2, // Use beam search
		MaxTokens:   5,
		Temperature: 1.0,
	}

	ctx := context.Background()
	segments, err := decoder.Decode(ctx, encoderOut, params)
	if err != nil {
		t.Fatalf("Decode with beam search returned error: %v", err)
	}

	if len(segments) == 0 {
		t.Error("expected at least one segment from beam search")
	}
}

func TestTokensToSegments_DecodesText(t *testing.T) {
	decoder := buildTestDecoder()

	v, err := vocab.New(
		[]string{"a", "b", "<|endoftext|>"},
		[]uint32{0, 0, 1},
	)
	if err != nil {
		t.Fatalf("vocab.New error: %v", err)
	}
	decoder.SetVocabulary(v)

	segments, err := decoder.tokensToSegments([]int32{99, 0, 1}, []int32{99})
	if err != nil {
		t.Fatalf("tokensToSegments error: %v", err)
	}
	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segments))
	}
	if segments[0].Text != "ab" {
		t.Fatalf("segment text = %q, want %q", segments[0].Text, "ab")
	}
	if len(segments[0].Tokens) != 2 {
		t.Fatalf("expected 2 tokens, got %d", len(segments[0].Tokens))
	}
	if segments[0].Tokens[0].Text != "a" || segments[0].Tokens[1].Text != "b" {
		t.Fatalf("unexpected token texts: %+v", segments[0].Tokens)
	}
}
