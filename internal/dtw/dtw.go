package dtw

import (
	"context"
	"fmt"
	"math"
	"strings"

	"github.com/whispergo/whisper.go/internal/model"
)

// Aligner performs DTW-based token alignment.
type Aligner struct {
	preset Preset
}

// New creates an aligner for the given model preset.
func New(modelName string) (*Aligner, error) {
	if modelName == "" {
		return nil, fmt.Errorf("dtw: model name cannot be empty")
	}

	preset, ok := GetPreset(modelName)
	if !ok {
		availableModels := getAvailableModels()
		return nil, fmt.Errorf("dtw: preset not found for model %q. Available presets: %s", modelName, availableModels)
	}

	return &Aligner{preset: preset}, nil
}

// Align computes per-token timestamps from attention weights.
// attention: [n_tokens, n_mel_frames], logits: [n_tokens, n_vocab]
// Returns TokenData with T0, T1 set in milliseconds.
func (a *Aligner) Align(ctx context.Context, attention [][]float32, logits [][]float32) ([]model.TokenData, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if len(attention) == 0 {
		return nil, fmt.Errorf("dtw: empty attention weights")
	}

	nTokens := len(attention)
	nFrames := len(attention[0])

	if nFrames == 0 {
		return nil, fmt.Errorf("dtw: empty attention frames")
	}

	// Verify logits shape
	if len(logits) != nTokens {
		return nil, fmt.Errorf("dtw: attention and logits token count mismatch: %d vs %d", nTokens, len(logits))
	}

	// Compute cost matrix: -log(attention)
	cost := make([][]float32, nTokens)
	for i := 0; i < nTokens; i++ {
		cost[i] = make([]float32, nFrames)
		for j := 0; j < nFrames; j++ {
			// Clip attention to avoid log(0)
			val := attention[i][j]
			if val < 1e-8 {
				val = 1e-8
			}
			cost[i][j] = float32(-math.Log(float64(val)))
		}
	}

	// Run DTW to find optimal path
	path, err := dtwAlign(cost)
	if err != nil {
		return nil, err
	}

	// Convert path to timestamps (10ms per frame)
	const msPerFrame = 10
	result := make([]model.TokenData, nTokens)

	for i := 0; i < nTokens; i++ {
		if i < len(path)-1 {
			startFrame := path[i]
			endFrame := path[i+1]

			// Ensure minimum duration of 10ms (1 frame)
			if endFrame <= startFrame {
				endFrame = startFrame + 1
			}

			result[i].T0 = int64(startFrame * msPerFrame)
			result[i].T1 = int64(endFrame * msPerFrame)
		} else {
			// Last token
			startFrame := path[i]
			endFrame := startFrame + 1
			result[i].T0 = int64(startFrame * msPerFrame)
			result[i].T1 = int64(endFrame * msPerFrame)
		}

		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
	}

	return result, nil
}

// dtwAlign computes the DTW path through the cost matrix using a simplified approach.
// Returns frame indices that align with tokens (0..nTokens) mapping to frame positions.
func dtwAlign(cost [][]float32) ([]int, error) {
	nTokens := len(cost)
	nFrames := len(cost[0])

	// Initialize DP table with infinity
	dp := make([][]float32, nTokens+1)
	for i := 0; i <= nTokens; i++ {
		dp[i] = make([]float32, nFrames+1)
		for j := 0; j <= nFrames; j++ {
			dp[i][j] = math.MaxFloat32
		}
	}
	dp[0][0] = 0

	// Forward pass: compute cumulative cost
	for i := 1; i <= nTokens; i++ {
		for j := 1; j <= nFrames; j++ {
			// Cost from three predecessors
			diagCost := dp[i-1][j-1]
			leftCost := dp[i][j-1]
			upCost := dp[i-1][j]

			minPrev := diagCost
			if leftCost < minPrev {
				minPrev = leftCost
			}
			if upCost < minPrev {
				minPrev = upCost
			}

			if minPrev == math.MaxFloat32 {
				dp[i][j] = math.MaxFloat32
			} else {
				dp[i][j] = cost[i-1][j-1] + minPrev
			}
		}
	}

	// Backtrack: find path from (nTokens, nFrames) to (0, 0)
	path := make([]int, nTokens+1)
	i, j := nTokens, nFrames

	for i > 0 || j > 0 {
		if i == 0 {
			j--
		} else if j == 0 {
			path[i] = 0
			i--
		} else {
			path[i] = j - 1
			diagVal := dp[i-1][j-1]
			leftVal := dp[i][j-1]
			upVal := dp[i-1][j]

			if diagVal <= leftVal && diagVal <= upVal {
				i--
				j--
			} else if leftVal <= upVal {
				j--
			} else {
				i--
			}
		}
	}
	path[0] = 0

	return path, nil
}

// getAvailableModels returns a comma-separated string of available model names.
func getAvailableModels() string {
	models := make([]string, 0, len(Presets))
	for name := range Presets {
		models = append(models, name)
	}
	return strings.Join(models, ", ")
}
