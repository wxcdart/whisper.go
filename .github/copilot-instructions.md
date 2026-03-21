# Copilot Instructions

## Project

`whisper.go` is a **Go-native port of [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)** — an automatic speech recognition (ASR) library implementing OpenAI's Whisper model. No CGo. No external C libraries. Tensor math, audio processing, and model inference are implemented in Go with SIMD acceleration via `gonum` BLAS.

## Build & Test

```sh
go build ./...
go test ./...

# Single test
go test ./internal/ml -run TestMatMul

# With race detector (recommended for concurrent encoder/decoder work)
go test -race ./...

# Lint
golangci-lint run
```

## Architecture

The port maps directly to whisper.cpp's pipeline. Every major layer below needs a Go equivalent.

### Inference pipeline (in order)

```
PCM f32 audio (16 kHz)
  → log-mel spectrogram       (internal/audio)
  → Whisper encoder           (internal/model)  — audio transformer, outputs cross-attn keys/values
  → Whisper decoder loop      (internal/model)  — text transformer, autoregressive token generation
  → token → text              (internal/vocab)
```

### Audio preprocessing (`internal/audio`)

- Input: raw PCM float32 at **16 000 Hz**
- Constants (must match exactly): `SampleRate=16000`, `NFFT=400`, `HopLength=160`, `ChunkSize=30` (seconds), `NMel=80`
- Compute Hann window → STFT frames → mel filterbank → log → normalise
- whisper.cpp reference: `log_mel_spectrogram()` in `src/whisper.cpp`

### Model / weight loading (`internal/gguf`)

- Models are distributed as **GGUF** files (superseded the older `.bin` format)
- GGUF is a self-describing binary format: magic + version + metadata key-value pairs + tensor descriptors + raw tensor data
- Reference parser: `ggml/src/gguf.cpp` and `ggml/include/gguf.h`
- Must support tensor dtypes: `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` (dequantise to F32 for compute unless you add quantised kernels)

### Tensor compute (`internal/ml`)

- whisper.cpp delegates all math to ggml (a C tensor library with lazy graph execution)
- For the Go port: implement eager ops — no need to replicate ggml's compute graph scheduler initially
- Required ops: `MatMul`, `Add`, `Mul`, `GELU`, `Softmax`, `LayerNorm`, `RoPE` (not used in Whisper), `Conv1D`, `Transpose`, `Reshape`, `Concat`, `MaskFill`
- Attention: scaled dot-product attention with causal mask for decoder self-attention; no mask for encoder; cross-attention uses encoder KV

### Encoder (`internal/model`)

Transformer encoder with a 1-D conv stem:

```
mel [NMel × T]
  Conv1(kernel=3, stride=1, pad=1) + GELU   → [n_state × T]
  Conv2(kernel=3, stride=2, pad=1) + GELU   → [n_state × T/2]
  + positional_embedding                     → [n_state × T/2]
  × N encoder blocks:
      LayerNorm → self-attention (no mask) → Add
      LayerNorm → MLP (Linear → GELU → Linear) → Add
  LayerNorm (post)
```

Tensor names in GGUF follow `encoder.conv1.weight`, `encoder.blocks.%d.attn.query.weight`, etc. (see `src/whisper-arch.h`).

### Decoder (`internal/model`)

Transformer decoder with cross-attention:

```
token_ids
  token_embedding + positional_embedding
  × N decoder blocks:
      LayerNorm → masked self-attention (causal) → Add   [KV cache grows here]
      LayerNorm → cross-attention (KV from encoder) → Add
      LayerNorm → MLP → Add
  LayerNorm
  @ token_embedding.weight^T  → logits [n_vocab]
```

Cross-attention KV is computed once per `Encode()` call and reused across all decode steps.

### Decoding strategies (`internal/model`)

- **Greedy**: `argmax` at each step (`best_of` restarts with temperature fallback)
- **Beam search**: maintain `beam_size` hypotheses; score = sum log-probs / length^α
- Temperature fallback: retry with `temperature += 0.2` when `compression_ratio > 2.4` or `avg_logprob < -1.0`

### Vocabulary / tokeniser (`internal/vocab`)

- Vocabulary is embedded in the GGUF metadata (not a separate file)
- Special tokens: `[SOT]`, `[EOT]`, `[BLANK]`, `[NO_SPEECH]`, timestamp tokens `<|0.00|>` … `<|30.00|>` (0.02 s steps)
- Language tokens and task tokens (`transcribe` / `translate`) are prepended to the decoder prompt

### Model hyperparameters (vary by model size)

| Model    | n_mels | n_audio_ctx | n_audio_state | n_audio_head | n_audio_layer | n_vocab | n_text_ctx | n_text_state | n_text_head | n_text_layer |
|----------|--------|-------------|---------------|--------------|---------------|---------|------------|--------------|-------------|--------------|
| tiny     | 80     | 1500        | 384           | 6            | 4             | 51865   | 448        | 384          | 6           | 4            |
| base     | 80     | 1500        | 512           | 8            | 6             | 51865   | 448        | 512          | 8           | 6            |
| small    | 80     | 1500        | 768           | 12           | 12            | 51865   | 448        | 768          | 12          | 12           |
| medium   | 80     | 1500        | 1024          | 16           | 24            | 51865   | 448        | 1024         | 16          | 24           |
| large    | 80     | 1500        | 1280          | 20           | 32            | 51866   | 448        | 1280         | 20          | 32           |

## Key Conventions

- **No CGo.** If a dependency pulls in CGo transitively, find an alternative.
- **Package layout**: one package per concern — `gguf` (parser), `audio` (mel), `ml` (tensors), `model` (encoder/decoder), `vocab` (tokeniser), public API at package root.
- **Tensor storage**: row-major float32 slices with explicit shape `[]int`. Avoid `interface{}` tensors; use a concrete `Tensor` struct.
- **Numerical parity**: tests must compare output against whisper.cpp's reference output (log-mel values, logits) within `1e-4` tolerance before any optimisation.
- **Reference commit**: `ggml-org/whisper.cpp@30c5194` is the baseline for this port.

## Style Guide

Follow the [Uber Go Style Guide](https://github.com/uber-go/guide/blob/master/style.md). Key points:

- Group imports: stdlib / external / internal, separated by blank lines (`goimports` order)
- Prefer `var ErrFoo = errors.New(...)` package-level sentinel errors; wrap with `fmt.Errorf("...: %w", err)`
- Return errors rather than panicking; panics only for programmer errors (impossible states)
- Use `context.Context` as the first parameter of any function that does I/O or long compute
- Initialise structs with field names, never positional
- Avoid `init()`; prefer explicit initialisation functions
- Table-driven tests with `t.Run`; test file in the same package (white-box) or `_test` package (black-box API tests)
- Use `//nolint:rulename // reason` (inline, never file-wide) when a lint suppression is truly necessary

## Concurrency

- All long-running or parallelisable work (mel computation, encoder, decoder steps) runs in goroutines.
- Every goroutine receives a `context.Context`; check `ctx.Err()` at the top of loops and before blocking calls.
- Use `errgroup.Group` (`golang.org/x/sync/errgroup`) to fan out goroutines and collect the first error:

  ```go
  g, ctx := errgroup.WithContext(ctx)
  for i := range workers {
      i := i
      g.Go(func() error {
          return process(ctx, i)
      })
  }
  if err := g.Wait(); err != nil { ... }
  ```

- Never start a goroutine without a way to stop it — always pass `ctx` and return when `ctx.Done()` is closed.
- Use `sync.WaitGroup` only when error propagation is not needed; prefer `errgroup` otherwise.
- Communicate results over channels; protect shared mutable state with `sync.Mutex` or `sync.RWMutex`, never raw memory sharing.
- The mel spectrogram worker threads (see `log_mel_spectrogram` in whisper.cpp) map naturally to a worker-pool pattern: split frames across `runtime.NumCPU()` goroutines via a channel of frame indices.

## SOLID Design Principles

- **Single Responsibility**: each package and type does one thing. `audio` only computes mel spectrograms; `gguf` only parses files; `ml` only does tensor math.
- **Open/Closed**: extend behaviour through interfaces, not by modifying existing types. E.g. new quantisation formats implement a `Dequantiser` interface rather than adding switch cases in existing code.
- **Liskov Substitution**: any `Encoder` or `Decoder` implementation must be substitutable without the caller changing behaviour. Do not add methods to concrete types that violate the interface contract.
- **Interface Segregation**: keep interfaces small and focused. Prefer several narrow interfaces (`TokenSampler`, `BeamScorer`) over one fat `Decoder` interface.
- **Dependency Inversion**: high-level packages (`model`) depend on abstractions (`ml.Tensor`, `vocab.Vocabulary`), not on concrete implementations. Pass dependencies via constructor parameters, never use package-level globals.
