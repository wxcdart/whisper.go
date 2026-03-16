package output

import (
	"context"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// VTTFormatter writes WebVTT subtitles.
type VTTFormatter struct{}

func (f VTTFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	if _, err := fmt.Fprint(w, "WEBVTT\n\n"); err != nil {
		return err
	}
	for _, seg := range result.Segments {
		if _, err := fmt.Fprintf(w, "%s --> %s\n%s\n\n",
			msToVTT(seg.StartMs), msToVTT(seg.EndMs), seg.Text); err != nil {
			return err
		}
	}
	return nil
}
