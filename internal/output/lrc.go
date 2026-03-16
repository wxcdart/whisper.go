package output

import (
	"context"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// LRCFormatter writes LRC lyrics format with centisecond timestamps.
type LRCFormatter struct{}

func (f LRCFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	for _, seg := range result.Segments {
		if _, err := fmt.Fprintf(w, "[%s]%s\n", msToLRC(seg.StartMs), seg.Text); err != nil {
			return err
		}
	}
	return nil
}
