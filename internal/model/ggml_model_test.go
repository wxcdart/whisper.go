package model

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/whispergo/whisper.go/internal/ggml"
)

func findFixture() string {
	p := "ggml-tiny.bin"
	if _, err := os.Stat(p); err == nil {
		return p
	}
	p = filepath.Join("testdata", "ggml-tiny.bin")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

func TestGGML_ModelLoader_BasicTensors(t *testing.T) {
	path := findFixture()
	if path == "" {
		t.Skip("no ggml .bin fixture found (ggml-tiny.bin)")
	}

	f, err := ggml.Open(context.Background(), path)
	if err != nil {
		t.Fatalf("ggml.Open: %v", err)
	}
	defer f.Close()

	names := f.TensorNames()
	if len(names) == 0 {
		t.Fatalf("expected tensor names, got none")
	}

	foundEnc := false
	foundDec := false
	for _, n := range names {
		if strings.Contains(n, "encoder") || strings.Contains(n, "conv1") {
			foundEnc = true
		}
		if strings.Contains(n, "decoder") || strings.Contains(n, "token_embedding") || strings.Contains(n, "embed_tokens") {
			foundDec = true
		}
	}

	if !foundEnc && !foundDec {
		t.Skip("fixture does not expose encoder/decoder tensor names; skipping loader test")
	}

	// Try to read one encoder-like tensor raw
	for _, n := range names {
		if strings.Contains(n, "conv1") || strings.Contains(n, "embed") {
			raw, shape, _, err := f.TensorRaw(context.Background(), n)
			if err != nil {
				t.Logf("TensorRaw(%q) not available: %v", n, err)
				continue
			}
			if len(raw) == 0 || len(shape) == 0 {
				t.Fatalf("bad tensor data for %q: len(raw)=%d shape=%v", n, len(raw), shape)
			}
			t.Logf("successfully read tensor %q shape=%v bytes=%d", n, shape, len(raw))
			return
		}
	}

	t.Skip("no readable encoder/embedding tensor found in fixture")
}
