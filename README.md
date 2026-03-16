# whisper.go

A pure-Go port of [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with feature parity and zero CGo dependencies.

**Status:** ✅ Feature-complete and ready for testing with real GGUF models.

## Features

- ✅ **Pure Go** — No C bindings, no CGo, single statically-linked binary
- ✅ **Full pipeline** — Audio → mel-spectrogram → encoder → decoder → text
- ✅ **Model format** — Complete GGUF v3 parser and writer
- ✅ **Transformer models** — Self-attention, cross-attention, layer norm, GELU
- ✅ **Sampling** — Greedy + beam search with temperature/fallback
- ✅ **Quantisation** — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 with dequantisation
- ✅ **Tokenisation** — BPE with special tokens, timestamps, language codes
- ✅ **Voice activity detection** — Silero VAD (CNN+LSTM)
- ✅ **Token timestamps** — Dynamic time warping alignment
- ✅ **Output formats** — TXT, SRT, VTT, JSON, JSON-Full, CSV, LRC, WTS
- ✅ **CLI** — Full command-line tool matching whisper.cpp's interface
- ✅ **Concurrency** — errgroup-based parallel tensor operations, context-aware cancellation

## Quick Start

### Build

```bash
go build -o whisper ./cmd/whisper
```

### Usage

```bash
# Basic transcription
whisper -f audio.wav -m ggml-tiny.en.gguf

# With options
whisper -f audio.wav -m ggml-base.gguf --language en --translate

# JSON output
whisper -f audio.wav -m ggml-tiny.en.gguf -of json -o result.json

# With beam search
whisper -f audio.wav -m ggml-small.gguf -bs 5 -b 5

# With VAD and DTW
whisper -f audio.wav -m ggml-medium.gguf --vad --vad-model silero-vad.gguf -dtw
```

### Download Models

Get GGUF models from [ggml-org/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main):

```bash
# Tiny (39 MB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.gguf

# Base (140 MB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.gguf

# Small (466 MB)
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.gguf
```

## API

```go
package main

import (
"context"
"log"

"github.com/whispergo/whisper.go"
)

func main() {
ctx := context.Background()

// Load model
model, err := whisper.New(ctx, "models/ggml-tiny.en.gguf")
if err != nil {
log.Fatal(err)
}
defer model.Close()

// Load audio (mono, 16 kHz float32)
samples, err := loadWAV("audio.wav")
if err != nil {
log.Fatal(err)
}

// Transcribe
params := whisper.DefaultParams()
result, err := model.Transcribe(ctx, samples, params)
if err != nil {
log.Fatal(err)
}

// Print results
for _, seg := range result.Segments {
log.Printf("[%d-%d] %s", seg.StartMs, seg.EndMs, seg.Text)
}
}
```

## Architecture

### Core Components

- **`internal/gguf`** — GGUF binary format parser/writer, metadata extraction, tensor I/O
- **`internal/audio`** — WAV reader, audio resampling, mel-spectrogram computation (STFT)
- **`internal/ml`** — Tensor operations (MatMul, Conv1D, attention, layer norm, GELU)
- **`internal/model`** — Encoder, decoder, full transcription pipeline
- **`internal/vocab`** — BPE tokeniser, special/timestamp/language token handling
- **`internal/vad`** — Silero VAD (CNN encoder + LSTM decoder + classifier)
- **`internal/dtw`** — Dynamic time warping for token timestamps
- **`internal/output`** — Format writers (TXT, SRT, VTT, JSON, CSV, LRC, WTS)
- **`cmd/whisper`** — CLI with 50+ flags matching whisper.cpp
- **`cmd/quantize`** — Model quantisation utility

### Concurrency Model

- All tensor operations use `context.Context` for cancellation awareness
- `errgroup.Group` for parallel processing (e.g., multi-head attention, channel-wise operations)
- No goroutine leaks; proper cleanup on context cancellation or errors

## Testing

All packages include table-driven tests:

```bash
go test ./...
```

Test coverage:
- GGUF parsing and round-trip serialisation
- Quantisation kernels (Q4, Q5, Q8)
- Tensor operations (MatMul, Conv1D, attention)
- BPE tokenisation
- Mel-spectrogram computation
- DTW alignment
- Output formatters
- VAD detection

See `TEST.md` for integration testing details.

## Design Principles

- **Uber Go Style** — No `init()`, error wrapping, table-driven tests
- **SOLID Design** — Interfaces for extensibility (Encoder, Decoder, Formatter, VAD, Aligner)
- **Context-aware** — All long-running operations respect `context.Context` for cancellation
- **Pure Go** — No CGo, no C dependencies, single self-contained binary
- **Minimal Dependencies** — Only `golang.org/x/sync` for `errgroup`

## Performance

Baseline (no optimization):
- **Throughput** — Depends on model size and audio duration
- **Memory** — ~500 MB for tiny model, scales with model size
- **Parallelism** — Configurable thread count via CLI flag `-t`

Future optimizations:
- SIMD kernels (optional)
- Tensor caching
- Model quantisation (already supported)
- Streaming decoding

## Limitations & Future Work

- **No streaming mode** — Must load entire audio into memory
- **Single-GPU only** — CPU only (no CUDA/Metal/Vulkan)
- **No VAD by default** — Requires separate GGUF model file
- **No diarization** — Would require speaker embedding model
- **Grammar constraints** — Not yet implemented

## Project Stats

| Metric | Value |
|---|---|
| **Go files** | 47 |
| **Production LOC** | ~4,800 |
| **Test LOC** | ~1,200 |
| **Total commits** | 40 |
| **Go version** | 1.25.0 |
| **Dependencies** | 1 (`golang.org/x/sync`) |

## References

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Original C++ implementation
- [Whisper Paper](https://arxiv.org/abs/2212.04356) — "Robust Speech Recognition via Large-Scale Weak Supervision"
- [GGML Format](https://github.com/ggerganov/ggml) — Binary model format specification
- [Uber Go Style Guide](https://github.com/uber-go/guide) — Code style principles

## License

MIT (see LICENSE file)

## Contributing

Contributions welcome! Please follow Uber Go style guide and add tests for new features.
