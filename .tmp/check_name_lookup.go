package main

import (
	"context"
	"fmt"
	"os"

	"github.com/whispergo/whisper.go/internal/ggml"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: check_name_lookup <model>")
		os.Exit(2)
	}
	p := os.Args[1]
	f, err := ggml.Open(context.Background(), p)
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	names := []string{"encoder.conv1.bias", "encoder.conv1.bia\ns"}
	for _, n := range names {
		qt, ok := f.TensorType(n)
		fmt.Printf("query %q -> ok=%v type=%v\n", n, ok, qt)
	}

	all := f.TensorNames()
	for i, n := range all {
		if i < 60 {
			fmt.Printf("%04d %q\n", i, n)
		}
	}
}
