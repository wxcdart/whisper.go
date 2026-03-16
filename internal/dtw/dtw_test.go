package dtw

import (
	"context"
	"math"
	"testing"
)

// TestAlign_OutputTimestamps verifies that timestamps are monotonic and properly spaced.
func TestAlign_OutputTimestamps(t *testing.T) {
	aligner, err := New("tiny.en")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Create synthetic attention matrix: 5 tokens, 50 mel frames
	nTokens := 5
	nFrames := 50
	attention := make([][]float32, nTokens)
	logits := make([][]float32, nTokens)

	for i := 0; i < nTokens; i++ {
		attention[i] = make([]float32, nFrames)
		logits[i] = make([]float32, 1000) // dummy logits

		// Create a synthetic attention pattern:
		// Token i focuses on frames around (i+1)*10
		centerFrame := (i + 1) * 10
		for j := 0; j < nFrames; j++ {
			// Gaussian-like attention around the center
			dist := float32(j - centerFrame)
			attention[i][j] = float32(0.9) * expf(-dist*dist/100)
			// Ensure non-zero values
			if attention[i][j] < 0.01 {
				attention[i][j] = 0.01
			}
		}
	}

	result, err := aligner.Align(context.Background(), attention, logits)
	if err != nil {
		t.Fatalf("Align failed: %v", err)
	}

	if len(result) != nTokens {
		t.Fatalf("Expected %d tokens, got %d", nTokens, len(result))
	}

	// Verify T0 < T1 for all tokens
	for i, token := range result {
		if token.T0 >= token.T1 {
			t.Errorf("Token %d: T0 (%d) >= T1 (%d)", i, token.T0, token.T1)
		}
		if token.T1-token.T0 < 10 {
			t.Errorf("Token %d: duration (%d ms) is less than 10ms", i, token.T1-token.T0)
		}
	}

	// Verify timestamps are monotonically increasing
	for i := 1; i < len(result); i++ {
		if result[i].T0 < result[i-1].T0 {
			t.Errorf("Token %d start time (%d) is before token %d start time (%d)",
				i, result[i].T0, i-1, result[i-1].T0)
		}
	}
}

// TestAlign_ContextCancellation verifies that context cancellation propagates.
func TestAlign_ContextCancellation(t *testing.T) {
	aligner, err := New("base")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	attention := [][]float32{
		{0.1, 0.2, 0.7},
		{0.3, 0.5, 0.2},
	}
	logits := [][]float32{
		make([]float32, 100),
		make([]float32, 100),
	}

	_, err = aligner.Align(ctx, attention, logits)
	if err == nil {
		t.Error("Expected cancellation error, got nil")
	}
	if err.Error() != "context canceled" {
		t.Errorf("Expected context canceled error, got: %v", err)
	}
}

// TestNew_InvalidPreset verifies error handling for unknown model names.
func TestNew_InvalidPreset(t *testing.T) {
	_, err := New("nonexistent.model")
	if err == nil {
		t.Fatal("Expected error for invalid preset, got nil")
	}
	if !contains(err.Error(), "preset not found") {
		t.Errorf("Error message should mention preset: %v", err)
	}
	if !contains(err.Error(), "tiny.en") {
		t.Errorf("Error should list available presets like tiny.en, got: %v", err)
	}
}

// TestNew_EmptyModelName verifies error for empty model name.
func TestNew_EmptyModelName(t *testing.T) {
	_, err := New("")
	if err == nil {
		t.Fatal("Expected error for empty model name")
	}
}

// TestAlign_EmptyAttention verifies error handling for empty input.
func TestAlign_EmptyAttention(t *testing.T) {
	aligner, _ := New("tiny")
	ctx := context.Background()

	_, err := aligner.Align(ctx, [][]float32{}, [][]float32{})
	if err == nil {
		t.Fatal("Expected error for empty attention")
	}
}

// TestAlign_ShapeMismatch verifies logits/attention mismatch detection.
func TestAlign_ShapeMismatch(t *testing.T) {
	aligner, _ := New("tiny")
	ctx := context.Background()

	attention := [][]float32{
		{0.1, 0.9},
	}
	logits := [][]float32{
		make([]float32, 100),
		make([]float32, 100),
	}

	_, err := aligner.Align(ctx, attention, logits)
	if err == nil {
		t.Fatal("Expected error for shape mismatch")
	}
}

// Helper function to check if error message contains a substring.
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// expf computes e^x using Go's math package.
func expf(x float32) float32 {
	return float32(exp(float64(x)))
}

// exp is an approximation of e^x. For testing, we use the standard math package.
func exp(x float64) float64 {
	if x > 700 {
		return math.MaxFloat64
	}
	if x < -700 {
		return 0
	}

	// Taylor series approximation for small values is good enough for tests
	result := 1.0
	term := 1.0
	for i := 1; i < 20; i++ {
		term *= x / float64(i)
		result += term
		if abs(term) < 1e-15 {
			break
		}
	}
	return result
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
