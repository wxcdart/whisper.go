package dtw

import (
	"context"

	"github.com/whispergo/whisper.go/internal/ml"
)

// Preset names for alignment heads.
const (
	PresetTinyEN       = "tiny.en"
	PresetTiny         = "tiny"
	PresetBaseEN       = "base.en"
	PresetBase         = "base"
	PresetSmallEN      = "small.en"
	PresetSmall        = "small"
	PresetMediumEN     = "medium.en"
	PresetMedium       = "medium"
	PresetLargeV1      = "large-v1"
	PresetLargeV2      = "large-v2"
	PresetLargeV3      = "large-v3"
	PresetLargeV3Turbo = "large-v3-turbo"
)

// Aligner maps attention weights to token timestamps.
type Aligner interface {
	// Align runs DTW over collected attention weights and returns per-token ms timestamps.
	Align(ctx context.Context, attnWeights []ml.Tensor, nAudioFrames int) (startMs, endMs []int64, err error)
}

// New returns an Aligner for the given preset name.
// preset may be one of the PresetXxx constants or empty (disabled).
func New(preset string) (Aligner, error) { panic("not implemented") }
