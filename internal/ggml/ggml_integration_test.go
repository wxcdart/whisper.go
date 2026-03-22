package ggml

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenLegacyBin_HasEncoderTensors(t *testing.T) {
	p := "ggml-tiny.bin"
	if _, err := os.Stat(p); os.IsNotExist(err) {
		p = filepath.Join("testdata", "ggml-tiny.bin")
	}
	if _, err := os.Stat(p); os.IsNotExist(err) {
		t.Skip("no ggml .bin fixture found (ggml-tiny.bin)")
	}

	f, err := Open(context.Background(), p)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer f.Close()

	want := []string{"encoder.conv1.bias", "encoder.conv1.weight"}
	for _, w := range want {
		if _, ok := f.TensorType(w); !ok {
			t.Fatalf("expected tensor %s present", w)
		}
	}
}
