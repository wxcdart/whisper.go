package output

import (
	"context"
	"encoding/json"
	"io"

	"github.com/whispergo/whisper.go/internal/model"
)

// JSONFormatter writes a compact JSON transcription (no token detail).
type JSONFormatter struct{}

// JSONFullFormatter writes JSON including per-token metadata.
type JSONFullFormatter struct{}

type jsonSegment struct {
	From string `json:"from"`
	To   string `json:"to"`
	Text string `json:"text"`
}

type jsonFullSegment struct {
	From   string      `json:"from"`
	To     string      `json:"to"`
	Text   string      `json:"text"`
	Tokens []jsonToken `json:"tokens"`
}

type jsonToken struct {
	ID    int32   `json:"id"`
	P     float32 `json:"p"`
	PT    float32 `json:"pt"`
	PTSum float32 `json:"ptsum"`
	T0    int64   `json:"t0"`
	T1    int64   `json:"t1"`
	Text  string  `json:"text"`
}

type jsonResult struct {
	Transcription []jsonSegment `json:"transcription"`
}

type jsonFullResult struct {
	Transcription []jsonFullSegment `json:"transcription"`
}

func (f JSONFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	segs := make([]jsonSegment, len(result.Segments))
	for i, s := range result.Segments {
		segs[i] = jsonSegment{
			From: msToVTT(s.StartMs),
			To:   msToVTT(s.EndMs),
			Text: s.Text,
		}
	}
	return json.NewEncoder(w).Encode(jsonResult{Transcription: segs})
}

func (f JSONFullFormatter) Format(_ context.Context, w io.Writer, result *model.Result, _ Options) error {
	segs := make([]jsonFullSegment, len(result.Segments))
	for i, s := range result.Segments {
		tokens := make([]jsonToken, len(s.Tokens))
		for j, t := range s.Tokens {
			tokens[j] = jsonToken{
				ID:    t.ID,
				P:     t.P,
				PT:    t.PT,
				PTSum: t.PTSum,
				T0:    t.T0,
				T1:    t.T1,
				Text:  t.Text,
			}
		}
		segs[i] = jsonFullSegment{
			From:   msToVTT(s.StartMs),
			To:     msToVTT(s.EndMs),
			Text:   s.Text,
			Tokens: tokens,
		}
	}
	return json.NewEncoder(w).Encode(jsonFullResult{Transcription: segs})
}
