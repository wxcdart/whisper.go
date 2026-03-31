package main

import (
	"context"
	"fmt"

	"github.com/whispergo/whisper.go/internal/ggml"
)

func main() {
	for _, path := range []string{
		"models/oxide-whisper-tiny-q4_0.gguf",
		"models/whisper-small-q4_0.gguf",
		"models/whisper-small-q4_0-whispercpp.gguf",
		"models/ggml-tiny.bin",
	} {
		f, err := ggml.Open(context.Background(), path)
		if err != nil {
			fmt.Printf("%s err: %v\n", path, err)
			continue
		}
		_, shape, err := f.Tensor(context.Background(), "mel_filters")
		fmt.Printf("%s mel_filters shape=%v err=%v\n", path, shape, err)
		_ = f.Close()
	}
}
