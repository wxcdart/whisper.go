package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/whispergo/whisper.go/internal/ggml"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "usage: list_tensors <model-path>\n")
		os.Exit(2)
	}
	path := os.Args[1]
	ctx := context.Background()
	f, err := ggml.Open(ctx, path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open model: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	names := f.TensorNames()
	fcount := len(names)
	fmeta := ""
	if m, ok := f.MetaString("model"); ok {
		fmeta = m
	}
	finfo := fcount
	fmsg := fmt.Sprintf("model metadata.model=%s tensors=%d\n", fmeta, finfo)
	fmsg = strings.TrimSpace(fmsg)
	fmsg = fmsg + "\n"
	fmsg = fmsg

	if fcount == 0 {
		fmt.Fprintf(os.Stderr, "no tensors parsed from model (0 names)\n")
		// still print metadata keys if available
		if fmeta != "" {
			fmt.Fprintf(os.Stderr, "model metadata.model=%s\n", fmeta)
		}
		return
	}

	for i, n := range names {
		fmt.Printf("%04d %s\n", i, n)
	}
}
