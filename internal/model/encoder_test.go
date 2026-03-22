package model

import (
	"context"
	"errors"
	"testing"

	"github.com/whispergo/whisper.go/internal/ml"
)

const (
	testNMels       = 80
	testNAudioState = 4
	testNHead       = 2
	testNAudioLayer = 1
	testNAudioCtx   = 1500
	testHeadDim     = testNAudioState / testNHead // 2
	testT           = 8
)

// buildTestEncoder constructs a WhisperEncoder with tiny dimensions and small nonzero weights.
func buildTestEncoder() *WhisperEncoder {
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

	block := encoderBlock{
		attnLnW: ones(testNAudioState),
		attnLnB: zeros(testNAudioState),
		attn: selfAttn{
			qW:   small(testNAudioState, testNAudioState),
			qB:   zeros(testNAudioState),
			kW:   small(testNAudioState, testNAudioState),
			vW:   small(testNAudioState, testNAudioState),
			vB:   zeros(testNAudioState),
			outW: small(testNAudioState, testNAudioState),
			outB: zeros(testNAudioState),
		},
		mlpLnW: ones(testNAudioState),
		mlpLnB: zeros(testNAudioState),
		// mlp0W: [4D, D], mlp2W: [D, 4D]
		mlp0W: small(4*testNAudioState, testNAudioState),
		mlp0B: zeros(4 * testNAudioState),
		mlp2W: small(testNAudioState, 4*testNAudioState),
		mlp2B: zeros(testNAudioState),
	}

	return &WhisperEncoder{
		nMels:       testNMels,
		nAudioState: testNAudioState,
		nHead:       testNHead,
		nAudioLayer: testNAudioLayer,
		nAudioCtx:   testNAudioCtx,
		// conv1W: [nAudioState, nMels, 3]; conv2W: [nAudioState, nAudioState, 3]
		conv1W:  small(testNAudioState, testNMels, 3),
		conv1B:  zeros(testNAudioState),
		conv2W:  small(testNAudioState, testNAudioState, 3),
		conv2B:  zeros(testNAudioState),
		posEmb:  ml.New(testNAudioCtx, testNAudioState),
		blocks:  []encoderBlock{block},
		lnPostW: ones(testNAudioState),
		lnPostB: zeros(testNAudioState),
	}
}

// TestEncode_OutputShape verifies that Encode returns the correct shape [T/2, nAudioState].
func TestEncode_OutputShape(t *testing.T) {
	enc := buildTestEncoder()
	mel := ml.New(testNMels, testT)

	out, err := enc.Encode(context.Background(), mel)
	if err != nil {
		t.Fatalf("Encode returned unexpected error: %v", err)
	}

	// Conv2 with stride=2 produces T'' = (T + 2*1 - 3)/2 + 1 = T/2 for even T.
	wantT := testT / 2
	if len(out.Shape) != 2 || out.Shape[0] != wantT || out.Shape[1] != testNAudioState {
		t.Errorf("output shape = %v; want [%d, %d]", out.Shape, wantT, testNAudioState)
	}
}

// TestEncode_ContextCancellation verifies that a cancelled context is respected.
func TestEncode_ContextCancellation(t *testing.T) {
	enc := buildTestEncoder()
	mel := ml.New(testNMels, testT)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately before Encode

	_, err := enc.Encode(ctx, mel)
	if err == nil {
		t.Fatal("Encode should have returned an error for cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled in error chain; got: %v", err)
	}
}
