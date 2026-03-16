package output_test

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/whispergo/whisper.go/internal/model"
	"github.com/whispergo/whisper.go/internal/output"
)

var testResult = &model.Result{
	Language: "en",
	Segments: []model.Segment{
		{
			StartMs: 0,
			EndMs:   2000,
			Text:    "Hello world",
			Tokens: []model.TokenData{
				{ID: 1, Text: "Hello", P: 0.9, T0: 0, T1: 1000},
				{ID: 2, Text: " world", P: 0.8, T0: 1000, T1: 2000},
			},
		},
		{
			StartMs: 2000,
			EndMs:   4500,
			Text:    "Foo bar",
			Tokens: []model.TokenData{
				{ID: 3, Text: "Foo", P: 0.95, T0: 2000, T1: 3000},
				{ID: 4, Text: " bar", P: 0.85, T0: 3000, T1: 4500},
			},
		},
	},
}

func mustFormat(t *testing.T, name string) output.Formatter {
	t.Helper()
	f, err := output.Format(name)
	if err != nil {
		t.Fatalf("Format(%q): %v", name, err)
	}
	return f
}

func run(t *testing.T, f output.Formatter, opts output.Options) string {
	t.Helper()
	var buf bytes.Buffer
	if err := f.Format(context.Background(), &buf, testResult, opts); err != nil {
		t.Fatalf("Format: %v", err)
	}
	return buf.String()
}

func TestFormat_UnknownName(t *testing.T) {
	_, err := output.Format("unknown")
	if err == nil {
		t.Fatal("expected error for unknown format")
	}
}

func TestTxtFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "txt"), output.Options{})
	if !strings.Contains(got, "Hello world") {
		t.Errorf("missing text: %q", got)
	}
	if !strings.Contains(got, "-->") {
		t.Errorf("expected timestamps: %q", got)
	}
}

func TestTxtFormatter_NoTimestamps(t *testing.T) {
	got := run(t, mustFormat(t, "txt"), output.Options{NoTimestamps: true})
	if strings.Contains(got, "-->") {
		t.Errorf("unexpected timestamps: %q", got)
	}
	if !strings.Contains(got, "Hello world") {
		t.Errorf("missing text: %q", got)
	}
}

func TestSRTFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "srt"), output.Options{})
	if !strings.Contains(got, "1\n") {
		t.Errorf("missing sequence number: %q", got)
	}
	if !strings.Contains(got, "00:00:00,000 --> 00:00:02,000") {
		t.Errorf("unexpected SRT timestamp: %q", got)
	}
	if !strings.Contains(got, "Hello world") {
		t.Errorf("missing text: %q", got)
	}
}

func TestVTTFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "vtt"), output.Options{})
	if !strings.HasPrefix(got, "WEBVTT\n") {
		t.Errorf("missing WEBVTT header: %q", got)
	}
	if !strings.Contains(got, "00:00:00.000 --> 00:00:02.000") {
		t.Errorf("unexpected VTT timestamp: %q", got)
	}
}

func TestJSONFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "json"), output.Options{})
	if !strings.Contains(got, `"transcription"`) {
		t.Errorf("missing transcription key: %q", got)
	}
	if !strings.Contains(got, "Hello world") {
		t.Errorf("missing text: %q", got)
	}
	if strings.Contains(got, `"tokens"`) {
		t.Errorf("json (non-full) should not contain tokens: %q", got)
	}
}

func TestJSONFullFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "json-full"), output.Options{})
	if !strings.Contains(got, `"transcription"`) {
		t.Errorf("missing transcription key: %q", got)
	}
	if !strings.Contains(got, `"tokens"`) {
		t.Errorf("missing tokens key: %q", got)
	}
}

func TestCSVFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "csv"), output.Options{})
	if !strings.HasPrefix(got, "start_ms,end_ms,text\n") {
		t.Errorf("missing CSV header: %q", got)
	}
	if !strings.Contains(got, "0,2000") {
		t.Errorf("missing first row: %q", got)
	}
}

func TestLRCFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "lrc"), output.Options{})
	if !strings.Contains(got, "[00:00.00]Hello world") {
		t.Errorf("unexpected LRC output: %q", got)
	}
}

func TestWTSFormatter(t *testing.T) {
	got := run(t, mustFormat(t, "wts"), output.Options{})
	if !strings.Contains(got, "drawtext=") {
		t.Errorf("missing drawtext: %q", got)
	}
	if !strings.Contains(got, "between(t,") {
		t.Errorf("missing between(): %q", got)
	}
}
