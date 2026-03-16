package output

import (
	"context"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// TxtFormatter writes plain-text transcription output.
type TxtFormatter struct{}

func (f TxtFormatter) Format(_ context.Context, w io.Writer, result *model.Result, opts Options) error {
	for _, seg := range result.Segments {
		var err error
		if opts.NoTimestamps {
			_, err = fmt.Fprintf(w, "%s\n", seg.Text)
		} else {
			_, err = fmt.Fprintf(w, "[%s --> %s]  %s\n", msToVTT(seg.StartMs), msToVTT(seg.EndMs), seg.Text)
		}
		if err != nil {
			return err
		}
	}
	return nil
}
