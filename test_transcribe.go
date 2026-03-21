//go:build ignore
// +build ignore

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/wxcdart/whisper.go"
	"github.com/wxcdart/whisper.go/internal/audio"
)

func main() {
	ctx := context.Background()

	fmt.Println("Loading audio...")
	start := time.Now()
	pcm, err := audio.ReadWavFile("testdata/test.wav")
	if err != nil {
		log.Fatalf("Error reading audio: %v", err)
	}
	fmt.Printf("Audio loaded in %v: %d samples\n", time.Since(start), len(pcm))

	fmt.Println("Loading model...")
	start = time.Now()
	m, err := whisper.New(ctx, "models/oxide-whisper-tiny-q4_0.gguf")
	if err != nil {
		log.Fatalf("Error loading model: %v", err)
	}
	fmt.Printf("Model loaded in %v\n", time.Since(start))

	fmt.Println("Transcribing...")
	start = time.Now()
	result, err := m.Transcribe(ctx, pcm, nil)
	if err != nil {
		log.Fatalf("Error transcribing: %v", err)
	}
	fmt.Printf("Transcription completed in %v\n", time.Since(start))

	fmt.Println("\nResult:")
	for i, seg := range result.Segments {
		fmt.Printf("[%d] %q\n", i, seg.Text)
	}
}
