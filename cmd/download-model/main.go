package main

import (
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

// GGUF format models from community repos
var ggufModels = map[string]string{
	// vonjack repo (large models)
	"whisper-large-v3-f16":  "vonjack/whisper-large-v3-gguf",
	"whisper-large-v3-q8_0": "vonjack/whisper-large-v3-gguf",
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
		fmt.Fprintf(os.Stderr, "  -o DIR     output directory (default: current directory)\n")
		fmt.Fprintf(os.Stderr, "  -l         list available models\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s whisper-large-v3-q8_0\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -o ./models whisper-large-v3-f16\n", os.Args[0])
		printModels()
		os.Exit(1)
	}

	model := args[0]

	if !isValidGGUFModel(model) {
		fmt.Fprintf(os.Stderr, "error: invalid GGUF model '%s'\n\n", model)
		printModels()
		os.Exit(1)
	}

	// Create output directory
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to create output directory: %v\n", err)
		os.Exit(1)
	}

	outPath := filepath.Join(*outDir, fmt.Sprintf("%s.gguf", model))

	// Check if file already exists
	if _, err := os.Stat(outPath); err == nil {
		fmt.Fprintf(os.Stderr, "Model already exists at %s\n", outPath)
		return
	}

	if err := downloadGGUFModel(model, outPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Successfully downloaded %s to %s\n", model, outPath)
	fmt.Printf("\nYou can now use it with:\n\n")
	fmt.Printf("  $ whisper -m %s -f audio.wav\n\n", outPath)
}

func isValidGGUFModel(model string) bool {
	_, ok := ggufModels[model]
	return ok
}

func printModels() {
	fmt.Printf("\nAvailable GGUF models (from community repos):\n\n")
	fmt.Printf("whisper.go supports GGUF models only.\n")
	fmt.Printf("Current available:\n\n")
	for model := range ggufModels {
		fmt.Printf("  %s\n", model)
	}
	fmt.Printf("\nSource: vonjack/whisper-large-v3-gguf\n\n")
}

func downloadGGUFModel(model, outPath string) error {
	// Determine which repo has this model
	repo, ok := ggufModels[model]
	if !ok {
		return fmt.Errorf("model %s not found in GGUF repos", model)
	}

	baseURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main", repo)
	url := fmt.Sprintf("%s/%s.gguf", baseURL, model)

	fmt.Printf("Downloading %s (GGUF format) from %s...\n", model, repo)

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status %d (model may not exist at %s)", resp.StatusCode, url)
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
