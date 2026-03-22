package main

import (
"context"
"fmt"

"github.com/whispergo/whisper.go/internal/ggml"
)

func main() {
ctx := context.Background()
f, err := ggml.Open(ctx, "models/ggml-tiny.bin")
if err != nil { panic(err) }
defer f.Close()

for _, n := range []string{
"encoder.conv1.weight",
"encoder.conv1.bias",
"encoder.conv2.weight",
"encoder.conv2.bias",
"encoder.positional_embedding",
"decoder.positional_embedding",
} {
_, s, err := f.Tensor(ctx, n)
fmt.Printf("%s => shape=%v err=%v\n", n, s, err)
}
}
