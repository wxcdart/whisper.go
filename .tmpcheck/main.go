package main

import (
	"context"
	"fmt"

	"github.com/whispergo/whisper.go/internal/gguf"
)

func main() {
	f, err := gguf.Open(context.Background(), "models/oxide-model-tiny-q80.gguf")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	raw, ok := f.Meta("tokenizer.ggml.tokens")
	if !ok {
		raw, ok = f.Meta("tokenizer.tokens")
	}
	if !ok {
		fmt.Println("no tokenizer token metadata")
		return
	}
	switch t := raw.(type) {
	case []string:
		fmt.Printf("type=[]string len=%d t[19639]=%q t[21300]=%q\n", len(t), t[19639], t[21300])
	case []any:
		fmt.Printf("type=[]any len=%d\n", len(t))
		if len(t) > 19639 {
			fmt.Printf("t[19639]=%T %v\n", t[19639], t[19639])
		}
		if len(t) > 21300 {
			fmt.Printf("t[21300]=%T %v\n", t[21300], t[21300])
		}
	default:
		fmt.Printf("type=%T\n", raw)
	}
}
