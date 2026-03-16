package output

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// CSVFormatter writes a CSV with columns start_ms, end_ms, text.
type CSVFormatter struct{}

func (f CSVFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	cw := csv.NewWriter(w)
	if err := cw.Write([]string{"start_ms", "end_ms", "text"}); err != nil {
		return err
	}
	for _, seg := range result.Segments {
		if err := cw.Write([]string{
			fmt.Sprintf("%d", seg.StartMs),
			fmt.Sprintf("%d", seg.EndMs),
			seg.Text,
		}); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
