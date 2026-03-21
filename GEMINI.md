# whisper.go — Go Whisper Implementation

`whisper.go` is a high-performance Go implementation of OpenAI's Whisper speech-to-text model. It uses the GGUF model format and aims for numerical parity and feature compatibility with `whisper.cpp`.

## Project Overview

- **Language:** Go 1.25.0
- **Model Format:** GGUF (including support for quantized types like `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`).
- **Architecture:** Go-native implementation of ML kernels and transcription pipeline (no CGo/C bindings), leveraging `errgroup` for parallelization and `gonum` BLAS SIMD kernels.
- **Key Features:**
    - GGUF parser and dequantizer.
    - Audio processing (WAV reading, Log-mel spectrogram).
    - Transformer Encoder and Decoder.
    - Greedy and Beam search decoding.
    - Voice Activity Detection (Silero VAD).
    - Token-level timestamps via DTW.
    - Multiple output formats (SRT, VTT, JSON, etc.).

## Directory Structure

- `cmd/`: CLI applications.
    - `whisper/`: Main transcription CLI.
    - `quantize/`: Model quantization tool.
- `internal/`: Core implementation packages.
    - `audio/`: WAV reading and Mel feature extraction.
    - `gguf/`: GGUF file parsing and tensor dequantization.
    - `ml/`: Tensor operations and mathematical kernels (MatMul, Conv1D, etc.).
    - `vocab/`: Tokenization and vocabulary management.
    - `model/`: Encoder, Decoder, and the full transcription pipeline.
    - `vad/`: Silero VAD implementation.
    - `dtw/`: Dynamic Time Warping for alignment.
    - `output/`: Formatter implementations for various transcript formats.
- `whisper.go`: The public Go API surface.

## Building and Running

The project uses a `Makefile` for standard development tasks.

- **Build:** `make build` (runs `go build ./...`)
- **Test:** `make test` (runs `go test -race ./...`)
- **Lint:** `make lint` (runs `golangci-lint run`)
- **Clean:** `make clean` (removes `build/`)

## Development Conventions

- **Numerical Parity:** All ML operations and outputs are validated against `whisper.cpp` reference fixtures.
    - Tolerance for `F32` operations: `1e-4`.
    - Tolerance for `F16` weights: `5e-3`.
- **Parallelism:** Parallelize compute-intensive operations (MatMul, Mel frames, Conv1D) using `golang.org/x/sync/errgroup` and `runtime.NumCPU()`.
- **Context Awareness:** All long-running or potentially blocking operations must accept and respect `context.Context`.
- **Git Workflow:** The project uses a "Wave" based development strategy documented in `TASKS.md`, encouraging the use of `git worktree` for parallel development of independent features.
- **Error Handling:** Use explicit error return values; panic only for truly unrecoverable programmer errors (e.g., incompatible tensor shapes in internal ops).

## Testing Strategy

- **Unit Tests:** Located alongside source files (e.g., `audio_test.go`, `gguf_test.go`).
- **Numerical Validation:** Tests often compare Go implementation results against pre-computed `jfk.wav` fixtures from `whisper.cpp`.
- **Snapshot Testing:** Output formatters are verified using snapshot tests against fixed `Result` objects.
