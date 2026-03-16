package output

import (
	"context"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// Options controls formatter behaviour.
type Options struct {
	NoTimestamps    bool
	PrintColors     bool
	PrintConfidence bool
	PrintSpecial    bool
	MaxLineLen      int
}

// Formatter writes a transcription result to an io.Writer.
type Formatter interface {
	Format(ctx context.Context, w io.Writer, result *model.Result, opts Options) error
}

// Format returns the appropriate Formatter for the given format name.
// Valid names: "txt", "srt", "vtt", "json", "json-full", "csv", "lrc", "wts".
func Format(name string) (Formatter, error) { panic("not implemented") }
