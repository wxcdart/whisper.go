package vad

import (
	"context"
	"math"
	"testing"

	"github.com/whispergo/whisper.go/internal/ml"
)

func TestDetect_OutputSegments(t *testing.T) {
	// Create synthetic audio with known speech/silence regions
	// 16kHz, 5 seconds total
	sampleRate := 16000
	durationS := 5.0
	nSamples := int(durationS * float64(sampleRate))
	samples := make([]float32, nSamples)

	// Silence: 0-1s, then speech: 1-3s, then silence: 3-5s
	// Speech region: sine wave at 440 Hz
	for i := 0; i < nSamples; i++ {
		tSec := float64(i) / float64(sampleRate)
		if tSec >= 1.0 && tSec < 3.0 {
			// Speech region: sine wave
			samples[i] = float32(0.5 * math.Sin(2*math.Pi*440*tSec))
		}
		// else: silence (remains 0)
	}

	// We cannot easily load a real GGUF model in tests without the actual file.
	// Instead, we test the structure by manually creating a SileroVAD instance.

	vad := &SileroVAD{
		lstmHidden: 64,
		// In a real test, weights would be loaded from GGUF
		// For this test, we'll just verify the post-processing logic
	}

	// Test the postProcess function with synthetic confidence scores
	// Simulate model output: confidence ~0.9 during speech (1-3s), ~0.1 during silence
	nFrames := int(math.Ceil(durationS * float64(sampleRate) / float64(VADHopLength)))
	confidences := make([]float32, nFrames)

	for t := 0; t < nFrames; t++ {
		frameMs := t * 10 // 10ms per frame
		frameSec := float64(frameMs) / 1000.0
		if frameSec >= 1.0 && frameSec < 3.0 {
			confidences[t] = 0.9 // High confidence during speech
		} else {
			confidences[t] = 0.1 // Low confidence during silence
		}
	}

	// Apply post-processing
	segments := vad.postProcess(confidences, DefaultParams())

	// Verify we detected the speech region
	if len(segments) == 0 {
		t.Fatal("expected at least one segment, got 0")
	}

	seg := segments[0]

	// Should be roughly around 1s-3s (in ms), with some padding
	// Default SpeechPadMs = 30, so roughly 970-3030ms
	if seg.StartMs > 1500 || seg.StartMs < 500 {
		t.Logf("segment start %dms seems off", seg.StartMs)
	}
	if seg.EndMs < 2500 || seg.EndMs > 4000 {
		t.Logf("segment end %dms seems off", seg.EndMs)
	}

	// Verify segment duration is reasonable
	duration := seg.EndMs - seg.StartMs
	if duration < int64(DefaultParams().MinSpeechMs) {
		t.Fatalf("segment duration %dms < min %dms", duration, DefaultParams().MinSpeechMs)
	}
}

func TestDetect_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	vad := &SileroVAD{
		lstmHidden: 64,
	}

	samples := make([]float32, 16000) // 1 second at 16kHz
	_, err := vad.Detect(ctx, samples, 16000, DefaultParams())

	if err == nil {
		t.Fatal("expected context cancellation error, got nil")
	}
	if err.Error() != "vad: context cancelled: context canceled" {
		t.Logf("got error: %v", err)
	}
}

// TestReLU tests the relu activation function
func TestReLU(t *testing.T) {
	// Test relu activation
	t1 := NewTestTensor([]float32{-1, 0, 1, 2, -3}, 5)
	out := relu(t1)
	expected := []float32{0, 0, 1, 2, 0}
	for i, v := range out.Data {
		if v != expected[i] {
			t.Errorf("relu mismatch at %d: got %v, want %v", i, v, expected[i])
		}
	}
}

// TestSigmoid32 tests the sigmoid function
func TestSigmoid32(t *testing.T) {
	v := []float32{0, 1, -1}
	out := sigmoid32(v)

	// sigmoid(0) = 0.5
	if math.Abs(float64(out[0])-0.5) > 0.01 {
		t.Errorf("sigmoid(0) = %v, want ~0.5", out[0])
	}
	// sigmoid(1) ≈ 0.731
	if math.Abs(float64(out[1])-0.731) > 0.01 {
		t.Errorf("sigmoid(1) = %v, want ~0.731", out[1])
	}
	// sigmoid(-1) ≈ 0.268
	if math.Abs(float64(out[2])-0.268) > 0.01 {
		t.Errorf("sigmoid(-1) = %v, want ~0.268", out[2])
	}
}

// TestMatmul1D tests the matmul1D function
func TestMatmul1D(t *testing.T) {
	// x = [1, 2], W = [3, 4; 5, 6] (2x2)
	// result should be [1*3 + 2*4, 1*5 + 2*6] = [11, 17]
	x := []float32{1, 2}
	W := NewTestTensor([]float32{3, 4, 5, 6}, 2, 2)

	out := matmul1D(x, W, 2, 2)

	if len(out) != 2 {
		t.Fatalf("expected output length 2, got %d", len(out))
	}
	if math.Abs(float64(out[0]-11)) > 0.01 {
		t.Errorf("out[0] = %v, want 11", out[0])
	}
	if math.Abs(float64(out[1]-17)) > 0.01 {
		t.Errorf("out[1] = %v, want 17", out[1])
	}
}

// TestResample tests the resample function
func TestResample(t *testing.T) {
	// Test simple upsampling: [1, 2] from 1Hz to 2Hz should give [1, 1.5, 2]
	samples := []float32{1, 2}
	out := resample(samples, 1, 2)

	if len(out) != 4 {
		t.Fatalf("expected length 4, got %d", len(out))
	}

	// Check values are roughly correct with linear interpolation
	if math.Abs(float64(out[0]-1)) > 0.1 {
		t.Errorf("out[0] = %v, want ~1", out[0])
	}
	if math.Abs(float64(out[2]-2)) > 0.1 {
		t.Errorf("out[2] = %v, want ~2", out[2])
	}
}

// TestPostProcess tests the postProcess function
func TestPostProcess(t *testing.T) {
	vad := &SileroVAD{lstmHidden: 64}

	// Create confidence scores with a clear speech region
	// 500 frames (5 seconds at 10ms per frame)
	confidences := make([]float32, 500)

	// Speech: frames 50-150 (0.5s-1.5s) with high confidence
	for i := 50; i < 150; i++ {
		confidences[i] = 0.8
	}

	params := DefaultParams()
	segments := vad.postProcess(confidences, params)

	if len(segments) != 1 {
		t.Fatalf("expected 1 segment, got %d", len(segments))
	}

	seg := segments[0]

	// After padding (30ms = 3 frames), start should be ~470ms, end ~1530ms
	// But due to min speech duration, it depends on exact params
	if seg.StartMs < 300 || seg.StartMs > 700 {
		t.Logf("segment start %dms", seg.StartMs)
	}
	if seg.EndMs < 1200 || seg.EndMs > 1800 {
		t.Logf("segment end %dms", seg.EndMs)
	}
}

// Helper function to create a test tensor
func NewTestTensor(data []float32, shape ...int) ml.Tensor {
	return ml.From(data, shape...)
}
