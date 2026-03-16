package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// Available models (from huggingface.co/ggerganov/whisper.cpp)
var models = []string{
	// Tiny
	"tiny", "tiny.en", "tiny-q5_1", "tiny.en-q5_1", "tiny-q8_0",
	// Base
	"base", "base.en", "base-q5_1", "base.en-q5_1", "base-q8_0",
	// Small
	"small", "small.en", "small.en-tdrz", "small-q5_1", "small.en-q5_1", "small-q8_0",
	// Medium
	"medium", "medium.en", "medium-q5_0", "medium.en-q5_0", "medium-q8_0",
	// Large
	"large-v1", "large-v2", "large-v2-q5_0", "large-v2-q8_0",
	"large-v3", "large-v3-q5_0", "large-v3-turbo", "large-v3-turbo-q5_0", "large-v3-turbo-q8_0",
}

func main() {
	outDir := flag.String("o", ".", "output directory for model file")
	listModels := flag.Bool("l", false, "list available models")
	flag.Parse()

	if *listModels {
		printModels()
		return
	}

	args := flag.Args()
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS] <model>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "OPTIONS:\n")
		fmt.Fprintf(os.Stderr, "  -o DIR    output directory (default: current directory)\n")
		fmt.Fprintf(os.Stderr, "  -l        list available models\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s tiny\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -o ./models base\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s small-q5_1\n", os.Args[0])
		printModels()
		os.Exit(1)
	}

	model := args[0]
	if !isValidModel(model) {
		fmt.Fprintf(os.Stderr, "error: invalid model '%s'\n\n", model)
		printModels()
		os.Exit(1)
	}

	// Create output directory
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to create output directory: %v\n", err)
		os.Exit(1)
	}

	outPath := filepath.Join(*outDir, fmt.Sprintf("ggml-%s.gguf", model))

	// Check if file already exists
	if _, err := os.Stat(outPath); err == nil {
		fmt.Fprintf(os.Stderr, "Model already exists at %s\n", outPath)
		return
	}

	// Download model
	if err := downloadModel(model, outPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Successfully downloaded %s to %s\n", model, outPath)
	fmt.Printf("\nYou can now use it with:\n\n")
	fmt.Printf("  $ whisper -m %s -f audio.wav\n\n", outPath)
}

func isValidModel(model string) bool {
	for _, m := range models {
		if m == model {
			return true
		}
	}
	return false
}

func printModels() {
	fmt.Printf("\nAvailable models:\n")
	currentClass := ""
	for _, model := range models {
		class := strings.FieldsFunc(model, func(r rune) bool {
			return r == '.' || r == '-'
		})[0]

		if class != currentClass {
			fmt.Printf("\n ")
			currentClass = class
		}
		fmt.Printf(" %s", model)
	}
	fmt.Printf("\n\n")
	fmt.Printf("Legend:\n")
	fmt.Printf("  .en           English-only model\n")
	fmt.Printf("  -q5_[01]      Quantized (smaller/faster)\n")
	fmt.Printf("  -tdrz         TinyDiarize (speaker detection)\n")
	fmt.Printf("\n")
}

func downloadModel(model, outPath string) error {
	// Determine source based on model
	baseURL := "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
	if strings.Contains(model, "tdrz") {
		baseURL = "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp/resolve/main"
	}

	url := fmt.Sprintf("%s/ggml-%s.gguf", baseURL, model)

	fmt.Printf("Downloading %s from %s...\n", model, baseURL)

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d", resp.StatusCode)
	}

	// Get content length for progress
	contentLength := resp.ContentLength
	if contentLength > 0 {
		fmt.Printf("Size: %.1f MB\n", float64(contentLength)/1024/1024)
	}

	// Create output file
	outFile, err := os.Create(outPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	// Copy with progress
	written, err := io.Copy(outFile, resp.Body)
	if err != nil {
		os.Remove(outPath)
		return fmt.Errorf("download interrupted: %w", err)
	}

	if contentLength > 0 && written != contentLength {
		os.Remove(outPath)
		return fmt.Errorf("incomplete download: expected %d bytes, got %d", contentLength, written)
	}

	return nil
}
