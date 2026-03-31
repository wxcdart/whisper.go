package main

import (
"context"
"fmt"
"reflect"

"github.com/whispergo/whisper.go/internal/ggml"
)

func probe(path string) {
ctx := context.Background()
f, err := ggml.Open(ctx, path)
if err != nil {
fmt.Printf("%s open err: %v\n", path, err)
return
}
defer f.Close()

fmt.Printf("\n== %s ==\n", path)
keys := []string{
"tokenizer.ggml.tokens",
"tokenizer.tokens",
"tokenizer.ggml.token_type",
"tokenizer.token_type",
"whisper.vocab.size",
"whisper.n_vocab",
}
for _, k := range keys {
v, ok := f.Meta(k)
if !ok {
fmt.Printf("%s: <missing>\n", k)
continue
}
fmt.Printf("%s: type=%s\n", k, reflect.TypeOf(v))
switch t := v.(type) {
case []string:
fmt.Printf("  len=%d first=%q\n", len(t), func() string { if len(t)>0 { return t[0] }; return "" }())
case []uint32:
fmt.Printf("  len=%d first=%d\n", len(t), func() uint32 { if len(t)>0 { return t[0] }; return 0 }())
}
}
}

func main() {
probe("models/oxide-whisper-tiny-q4_0.gguf")
probe("models/whisper-small-q4_0-whispercpp.gguf")
probe("models/whisper-small-q4_0.gguf")
}
