# whisper.go — Task Breakdown for Git Worktrees

Each wave lists branches that can be checked out and developed **simultaneously** as separate git worktrees.
A wave may only begin once all branches listed under its **"Requires merged"** line are merged into `main`.

---

## Wave 0 — Bootstrap (sequential, everyone blocks on this)

| Branch | Package(s) | Description |
|---|---|---|
| `feat/scaffold` | root, `cmd/whisper`, all `internal/*` | `go mod init`, directory skeleton, stub files, `.golangci.yml`, CI workflow |

**Setup:**
```sh
git init
git checkout -b feat/scaffold
# ... implement, then merge to main before starting Wave 1
```

---

## Wave 1 — Independent foundations  
_Requires merged: `feat/scaffold`_

Four branches with **zero inter-dependencies** — check them all out in parallel.

| Branch | Package(s) | Todo ID |
|---|---|---|
| `feat/gguf` | `internal/gguf` | `gguf` |
| `feat/audio` | `internal/audio` | `audio` |
| `feat/ml` | `internal/ml` | `ml` |
| `feat/quantize` | `cmd/quantize` | `quantize` ¹ |

¹ `feat/quantize` depends on `internal/gguf` but can be developed against the stub interface; merge after `feat/gguf` lands.

**Worktree setup:**
```sh
git worktree add ../whisper.go-gguf    -b feat/gguf
git worktree add ../whisper.go-audio   -b feat/audio
git worktree add ../whisper.go-ml      -b feat/ml
git worktree add ../whisper.go-quant   -b feat/quantize
```

### `feat/gguf` — GGUF Parser
- Parse magic, version, metadata KV map, tensor descriptors, raw tensor data
- Dequantise: `F32`, `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` → `[]float32`
- Public API: `gguf.Open(ctx, path) (*File, error)`
- Tests: round-trip metadata read, dequant values vs. known fixtures

### `feat/audio` — Audio Processing
- WAV reader: RIFF header, 8/16/24/32-bit int + 32-bit float samples; stereo → mono downmix; resample to 16 kHz
- Log-mel spectrogram: Hann window (`N_FFT=400`), STFT (`HopLength=160`), 80-band mel filterbank, log + normalise
- Parallelise frame computation via `errgroup` + `runtime.NumCPU()` workers
- Tests: mel output for `jfk.wav` vs. whisper.cpp reference fixtures (tolerance `1e-4`)

### `feat/ml` — Tensor / ML Ops
- `Tensor` struct: row-major `[]float32` + `shape []int`
- Ops: `MatMul` (batched + transposed), `Add`, `Mul`, `GELU`, `LayerNorm`, `Softmax`, `Conv1D` (stride 1 & 2), `Transpose`, `Reshape`, `View`, `Concat`, `ScaledDotProductAttention` (optional causal mask)
- Parallelise `MatMul` and `Conv1D` across rows with `errgroup`
- Tests: numerical parity vs. NumPy reference values

### `feat/quantize` — Quantize CLI Tool
- `cmd/quantize`: read source GGUF, re-quantise weight tensors, write output GGUF
- Supported target types: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`
- Depends on `internal/gguf`; develop against interface, merge after `feat/gguf`

---

## Wave 2 — Vocabulary + Output formats  
_Requires merged: `feat/gguf`, `feat/ml`_

| Branch | Package(s) | Todo ID | Requires |
|---|---|---|---|
| `feat/vocab` | `internal/vocab` | `vocab` | `feat/gguf` |
| `feat/output` | `internal/output` | `output` | `feat/vocab` ¹ |

¹ `feat/output` can stub `vocab.Token` and be mostly written before `feat/vocab` merges; only token-decode needs the real vocab.

```sh
git worktree add ../whisper.go-vocab   -b feat/vocab
git worktree add ../whisper.go-output  -b feat/output
```

### `feat/vocab` — Tokeniser
- Load tokens + token types from GGUF metadata
- `Token ↔ ID` maps
- Special tokens: `SOT`, `EOT`, `BLANK`, `NO_SPEECH`, `TRANSLATE`, `TRANSCRIBE`, `PREV`, `SOLM`, `NOT_SOT`
- Timestamp tokens `<|0.00|>` … `<|30.00|>` (1501 tokens, 0.02 s steps)
- 99 language tokens
- `Encode(text) []Token`, `Decode([]Token) string`

### `feat/output` — Output Formatters
- `Formatter` interface: `Write(ctx context.Context, w io.Writer, result *Result, opts Options) error`
- Implementations: `txt`, `srt`, `vtt`, `json`, `json-full`, `csv`, `lrc`, `wts`
- Stdout writer with ANSI colour (confidence-based) and `--print-confidence` support
- Tests: snapshot tests for each format against a fixed `Result` fixture

---

## Wave 3 — Encoder  
_Requires merged: `feat/ml`, `feat/vocab`_

| Branch | Package(s) | Todo ID |
|---|---|---|
| `feat/encoder` | `internal/model` (encoder only) | `encoder` |

```sh
git worktree add ../whisper.go-encoder -b feat/encoder
```

### `feat/encoder` — Audio Transformer Encoder
- `Encoder` interface: `Encode(ctx context.Context, mel Tensor) (Tensor, error)`
- Conv1 + Conv2 stem (GELU), sinusoidal positional embedding, N encoder blocks
- Each block: `LayerNorm → SelfAttn → residual; LayerNorm → MLP(GELU) → residual`
- Post LayerNorm; returns `[n_audio_state × T/2]` cross-KV tensor pair for decoder
- `ctx.Err()` checked between every block
- Load weights from `*gguf.File` by tensor name (`encoder.conv1.weight`, `encoder.blocks.%d.*`)
- Tests: encoder output norm vs. whisper.cpp reference for `tiny` model

---

## Wave 4 — Decoder, VAD, DTW  
_Requires merged: `feat/encoder`, `feat/audio`_  
_(all three branches are independent of each other)_

| Branch | Package(s) | Todo ID | Requires |
|---|---|---|---|
| `feat/decoder` | `internal/model` (decoder + greedy + beam) | `decoder-greedy`, `beam-search` | `feat/encoder` |
| `feat/vad` | `internal/vad` | `vad` | `feat/audio`, `feat/ml` |
| `feat/dtw` | `internal/dtw` | `dtw` | `feat/encoder` |

```sh
git worktree add ../whisper.go-decoder -b feat/decoder
git worktree add ../whisper.go-vad     -b feat/vad
git worktree add ../whisper.go-dtw     -b feat/dtw
```

### `feat/decoder` — Text Transformer Decoder
- Token + positional embeddings
- N decoder blocks: `LayerNorm → CausalSelfAttn(KV cache) → residual; LayerNorm → CrossAttn(encoderKV) → residual; LayerNorm → MLP → residual`
- Output projection: `@ token_embedding.weight^T → logits [n_vocab]`
- KV cache: pre-allocated `[n_layer × 2 × n_text_ctx × n_text_state]`
- **Greedy sampling**: `argmax`, temperature scaling, fallback retry (entropy / avg-logprob / no-speech thresholds), suppress-nst, suppress-regex
- **Beam search**: `beam_size` hypotheses, score = `Σlog_p / len^α`, prune per step, return best at `EOT`
- `Decoder` interface: `Decode(ctx context.Context, encoderOut Tensor, params DecoderParams) ([]Segment, error)`
- Tests: logit parity at step 0 for `tiny` model, known-good transcript for `jfk.wav`

### `feat/vad` — Silero VAD
- Load Silero VAD GGUF (separate model file via `--vad-model`)
- STFT frontend (`VAD_TENSOR_STFT_BASIS`), 4-layer conv encoder, LSTM decoder, final conv → sigmoid
- Return `[]SpeechSegment{StartMs, EndMs}`
- Post-process: pad segments, merge short silences, discard short segments
- All LSTM steps receive `context.Context`
- Tests: known speech/silence segments for test audio

### `feat/dtw` — DTW Token Timestamps
- Collect per-step attention weights from configured alignment heads during decode
- DTW alignment: text tokens ↔ mel frames → convert to seconds
- Write `TokenData.T0` / `T0` for each token
- Support all presets (`tiny.en`, `base`, `small`, `medium`, `large-v1/v2/v3`, `large-v3-turbo`)
- `DTW` interface: `Align(ctx context.Context, attnWeights []Tensor, nFrames int) ([]int64, error)`

---

## Wave 5 — Transcription Pipeline  
_Requires merged: `feat/decoder`, `feat/audio`_

| Branch | Package(s) | Todo ID |
|---|---|---|
| `feat/pipeline` | `internal/model` (pipeline) | `pipeline` |

```sh
git worktree add ../whisper.go-pipeline -b feat/pipeline
```

### `feat/pipeline` — Full Inference Loop
- 30-second chunk windowing with 1-second overlap
- Language detection (first-chunk logit sampling over language token set)
- Decoder prompt: `[SOT, lang, task, notimestamps?]` + `initial_prompt` tokens
- `--carry-initial-prompt`: prepend previous segment tokens to next window
- No-speech gate: `softmax(logits)[NO_SPEECH] > threshold` → skip segment
- Segment splitting by `--max-len` token count or `--split-on-word`
- `--offset-t` / `--duration` audio sub-range
- Multiple input files processed sequentially
- `Transcriber` interface: `Transcribe(ctx context.Context, audio []float32, params Params) (*Result, error)`

---

## Wave 6 — CLI (final integration)  
_Requires merged: `feat/pipeline`, `feat/beam-search` (in `feat/decoder`), `feat/output`, `feat/vad`, `feat/dtw`_

| Branch | Package(s) | Todo ID |
|---|---|---|
| `feat/cli` | `cmd/whisper` | `cli` |

```sh
git worktree add ../whisper.go-cli -b feat/cli
```

### `feat/cli` — CLI Binary
Wire all internal packages into a single `cobra`/`flag`-based binary. All flags must match `whisper.cpp`'s `cli` example exactly (same short and long names, same defaults).

**Flag groups:**

| Group | Flags |
|---|---|
| Input | `-f/--file`, `-m/--model` |
| Language | `-l/--language`, `-tr/--translate`, `-dl/--detect-language` |
| Inference | `-t/--threads`, `-p/--processors`, `-bo/--best-of`, `-bs/--beam-size`, `-tp/--temperature`, `-tpi/--temperature-inc`, `-et/--entropy-thold`, `-lpt/--logprob-thold`, `-nth/--no-speech-thold`, `-nf/--no-fallback`, `-ac/--audio-ctx` |
| Segmentation | `-mc/--max-context`, `-ml/--max-len`, `-sow/--split-on-word`, `-ot/--offset-t`, `-on/--offset-n`, `-d/--duration` |
| Output format | `-otxt`, `-osrt`, `-ovtt`, `-oj`, `-ojf`, `-ocsv`, `-olrc`, `-owts`, `-of/--output-file` |
| Display | `-nt/--no-timestamps`, `-ps/--print-special`, `-pc/--print-colors`, `--print-confidence`, `-pp/--print-progress`, `-np/--no-prints`, `-ls/--log-score` |
| DTW | `-dtw/--dtw` |
| Diarization | `-di/--diarize`, `-tdrz/--tinydiarize` |
| VAD | `--vad`, `-vm/--vad-model`, `-vt/--vad-threshold`, `-vspd`, `-vsd`, `-vmsd`, `-vp`, `-vo` |
| Grammar | `--grammar`, `--grammar-rule`, `--grammar-penalty` |
| Misc | `--prompt`, `--carry-initial-prompt`, `-sns/--suppress-nst`, `--suppress-regex`, `--debug-mode` |

---

## Dependency Graph (summary)

```
feat/scaffold
│
├── feat/gguf ──── feat/vocab ──── feat/output ──────────────────────┐
│       └───────────────────────── feat/quantize                     │
│                                                                     │
├── feat/audio ──────────────────────────────────────────────────────┤
│       └── feat/vad (+ feat/ml) ────────────────────────────────────┤
│                                                                     │
└── feat/ml ───── feat/encoder ─── feat/decoder ─── feat/pipeline ───┤
                        └───────── feat/dtw ─────────────────────────┘
                                                                      │
                                                             feat/cli ◄┘
```

## Numerical Validation (all waves)

Golden fixtures captured from `whisper.cpp@30c5194` using `jfk.wav`:
- **Mel**: first 100 frames, tolerance `1e-4`
- **Encoder out**: hidden state L2-norm, tolerance `1e-4`
- **Decoder step 0**: top-5 logit values + indices, tolerance `1e-4`
- **Transcript**: exact string match for `jfk.wav` → `" And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."`
- F16-loaded weights: tolerance `5e-3`
