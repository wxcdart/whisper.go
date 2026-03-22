package main

import (
	"context"
	"fmt"
	"os"

	"github.com/whispergo/whisper.go/internal/ggml"
)

func dump(path string) error {
	ctx := context.Background()
	f, err := ggml.Open(ctx, path)
	if err != nil {
		return err
	}
	defer f.Close()

	fmt.Printf("\n== %s ==\n", path)

	metaKeys := []string{
		"whisper.decoder.context_length",
		"whisper.decoder.embedding_length",
		"whisper.decoder.attention.head_count",
		"whisper.decoder.layer_count",
		"whisper.encoder.embedding_length",
		"whisper.vocab.size",
	}
	for _, key := range metaKeys {
		if v, ok := f.MetaUint32(key); ok {
			fmt.Printf("meta %-40s = %d\n", key, v)
		} else {
			fmt.Printf("meta %-40s = <missing>\n", key)
		}
	}

	tensorKeys := []string{
		"decoder.blocks.0.self_attn.query.weight",
		"decoder.blocks.0.attn.query.weight",
		"model.decoder.layers.0.self_attn.q_proj.weight",
		"decoder.positional_embedding",
		"model.decoder.embed_positions.weight",
		"decoder.token_embedding.weight",
		"model.decoder.embed_tokens.weight",
	}
	for _, key := range tensorKeys {
		if _, shape, err := f.Tensor(ctx, key); err == nil {
			fmt.Printf("tensor %-50s shape=%v\n", key, shape)
		}
	}

	return nil
}

func main() {
	paths := os.Args[1:]
	if len(paths) == 0 {
		paths = []string{
			"models/ggml-tiny.bin",
			"models/oxide-whisper-tiny-q4_0.gguf",
			"models/whisper-small-q4_0.gguf",
		}
	}

	for _, path := range paths {
		if err := dump(path); err != nil {
			fmt.Printf("err %s: %v\n", path, err)
		}
	}
}
