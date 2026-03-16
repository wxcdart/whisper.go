package output

import (
	"context"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// SRTFormatter writes SubRip (.srt) subtitles.
type SRTFormatter struct{}

func (f SRTFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	for i, seg := range result.Segments {
		if _, err := fmt.Fprintf(w, "%d\n%s --> %s\n%s\n\n",
			i+1, msToSRT(seg.StartMs), msToSRT(seg.EndMs), seg.Text); err != nil {
			return err
		}
	}
	return nil
}
