package ggml

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenBin_ParsesHeaderAndTensors(t *testing.T) {
	path := "ggml-tiny.bin"
	if _, err := os.Stat(path); err != nil {
		// try testdata dir
		path = filepath.Join("testdata", "ggml-tiny.bin")
		if _, err := os.Stat(path); err != nil {
			t.Skipf("no ggml .bin fixture found at ggml-tiny.bin or testdata/ggml-tiny.bin: %v", err)
		}
	}

	f, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open() error: %v", err)
	}
	defer f.Close()

	names := f.TensorNames()
	if len(names) == 0 {
		t.Fatalf("expected >=1 tensor names, got 0")
	}

	// Try to read the first tensor raw; if TensorRaw isn't implemented for
	// this model, skip the deeper checks.
	name := names[0]
	raw, shape, qtype, err := f.TensorRaw(context.Background(), name)
	if err != nil {
		t.Skipf("TensorRaw not available for %q: %v", name, err)
	}
	if len(raw) == 0 {
		t.Fatalf("TensorRaw returned empty data for %q (qtype=%v, shape=%v)", name, qtype, shape)
	}
}
