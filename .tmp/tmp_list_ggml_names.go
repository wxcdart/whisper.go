package main

import (
"context"
"fmt"
"github.com/whispergo/whisper.go/internal/ggml"
)

func main() {
f, err := ggml.Open(context.Background(), "models/ggml-tiny.bin")
if err != nil { panic(err) }
defer f.Close()
names := f.TensorNames()
fmt.Printf("count=%d\n", len(names))
for i, n := range names {
if i >= 20 { break }
fmt.Println(n)
}
}
