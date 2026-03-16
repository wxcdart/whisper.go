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

// Available old ggml format models (from huggingface.co/ggerganov/whisper.cpp)
var oldGGMLModels = []string{
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

// GGUF format models from community repos
var ggufModels = map[string]string{
	// vonjack repo (large models)
	"whisper-large-v3-f16":   "vonjack/whisper-large-v3-gguf",
	"whisper-large-v3-q8_0":  "vonjack/whisper-large-v3-gguf",
}

func main() {
	outDir := flag.String("o", ".", "output directory for model file")
	listModels := flag.Bool("l", false, "list available models")
	gguf := flag.Bool("gguf", false, "use GGUF format models (recommended)")
	flag.Parse()

	if *listModels {
		printModels(*gguf)
		return
	}

	args := flag.Args()
	if len(args) != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [OPTIONS] <model>\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "OPTIONS:\n")
		fmt.Fprintf(os.Stderr, "  -o DIR     output directory (default: current directory)\n")
		fmt.Fprintf(os.Stderr, "  -gguf      use GGUF format models (recommended, default: old ggml format)\n")
		fmt.Fprintf(os.Stderr, "  -l         list available models\n\n")
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  %s tiny                          # old ggml format\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -gguf whisper-large-v3-q8_0   # new GGUF format\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -o ./models base              # custom directory\n", os.Args[0])
		printModels(*gguf)
		os.Exit(1)
	}

	model := args[0]

	// Use appropriate validation and download function
	if *gguf {
		if !isValidGGUFModel(model) {
			fmt.Fprintf(os.Stderr, "error: invalid GGUF model '%s'\n\n", model)
			printModels(true)
			os.Exit(1)
		}
	} else {
		if !isValidOldModel(model) {
			fmt.Fprintf(os.Stderr, "error: invalid old ggml model '%s'\n\n", model)
			printModels(false)
			os.Exit(1)
		}
	}

	// Create output directory
	if err := os.MkdirAll(*outDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "error: failed to create output directory: %v\n", err)
		os.Exit(1)
	}

	var outPath string
	if *gguf {
		outPath = filepath.Join(*outDir, fmt.Sprintf("%s.gguf", model))
	} else {
		outPath = filepath.Join(*outDir, fmt.Sprintf("ggml-%s.bin", model))
	}

	// Check if file already exists
	if _, err := os.Stat(outPath); err == nil {
		fmt.Fprintf(os.Stderr, "Model already exists at %s\n", outPath)
		return
	}

	// Download model
	if *gguf {
		if err := downloadGGUFModel(model, outPath); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	} else {
		if err := downloadOldModel(model, outPath); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	}

	fmt.Printf("Successfully downloaded %s to %s\n", model, outPath)
	fmt.Printf("\nYou can now use it with:\n\n")
	fmt.Printf("  $ whisper -m %s -f audio.wav\n\n", outPath)
}

func isValidOldModel(model string) bool {
	for _, m := range oldGGMLModels {
		if m == model {
			return true
		}
	}
	return false
}

func isValidGGUFModel(model string) bool {
	_, ok := ggufModels[model]
	return ok
}

func printModels(gguf bool) {
	if gguf {
		fmt.Printf("\nAvailable GGUF models (from community repos):\n\n")
		fmt.Printf("GGUF models are the recommended format for whisper.go.\n")
		fmt.Printf("More models being added. Current available:\n\n")
		for model := range ggufModels {
			fmt.Printf("  %s\n", model)
		}
		fmt.Printf("\nSource: vonjack/whisper-large-v3-gguf\n\n")
	} else {
		fmt.Printf("\nAvailable old ggml format models (from ggerganov/whisper.cpp):\n")
		currentClass := ""
		for _, model := range oldGGMLModels {
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
		fmt.Printf("\nNote: Old ggml format (.bin) is not directly supported by whisper.go.\n")
		fmt.Printf("Use '--gguf' flag for GGUF models instead.\n\n")
	}
}

func downloadOldModel(model, outPath string) error {
	// Determine source based on model
	baseURL := "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

	url := fmt.Sprintf("%s/ggml-%s.bin", baseURL, model)

	fmt.Printf("Downloading %s (old ggml format) from %s...\n", model, baseURL)

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
