package output

import (
	"context"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// WTSFormatter writes an FFmpeg drawtext filter script for word-level highlighting.
type WTSFormatter struct{}

func (f WTSFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	for _, seg := range result.Segments {
		for _, tok := range seg.Tokens {
			start := msToSec(tok.T0)
			end := msToSec(tok.T1)
			_, err := fmt.Fprintf(w,
				"drawtext=fontfile=/path/to/font.ttf:text='%s':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,%.3f,%.3f)'\n",
				tok.Text, start, end,
			)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
