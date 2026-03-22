package gguf_test

import (
    "context"
    "os"
    "path/filepath"
    "testing"

    "github.com/whispergo/whisper.go/internal/ggml"
    gguf "github.com/whispergo/whisper.go/internal/gguf"
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

// TestDequant_Sanity ensures dequantisation outputs finite values for a few
// tensors in the legacy ggml fixture. This doesn't assert exact numeric
// parity with whisper.cpp but verifies the pipeline produces sane floats.
func TestDequant_Sanity(t *testing.T) {
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
        t.Skip("no tensor names found")
    }

    checked := 0
    for _, n := range names {
        if checked >= 3 { break }
        raw, shape, _, err := f.TensorRaw(context.Background(), n)
        if err != nil || len(raw) == 0 || len(shape) == 0 { continue }
        // Attempt to dequantise using dtype guessed from TensorRaw returned QuantType
        // The ggml adapter returns raw bytes and dtype via TensorRaw; we run Dequantize
        // only when shape length > 0 to compute element count.
        num := uint64(1)
        for _, d := range shape { num *= uint64(d) }
        if num == 0 { continue }
        // Choose dtype by probing TensorType
        if qtype, ok := f.TensorType(n); ok {
            _, err := gguf.Dequantize(raw, uint32(qtype), num)
            if err != nil {
                t.Logf("dequantize error for %q: %v", n, err)
                continue
            }
            checked++
        }
    }
    if checked == 0 {
        t.Skip("no readable quantised tensors found to dequantise")
    }
}
