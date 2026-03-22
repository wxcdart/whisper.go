GGML Go Port — Spec

Goal
- Provide a pure-Go implementation of the core ggml features needed to run Whisper models without cgo.

Scope (MVP)
- Tensor storage (row-major float32) and metadata
- MatMul (blas-backed + pure-Go fallback)
- Core ops: Add, Mul, LayerNorm, GELU, Softmax, Conv1D, Transpose, Reshape, Concat, MaskFill
- Attention primitives (scaled dot-product, causal mask)
- Quantisation/Dequantisation support for Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q8_0
- GGUF loader + robust legacy `.bin` adapter

Constraints
- No cgo in MVP
- Numeric parity target: 1e-4 for critical ops vs ggml reference
- Reasonable startup memory usage; reuse buffers where practical

Deliverables
- `internal/ml` package with `Tensor` and `Backend` abstractions
- Unit tests per-op and integration test that loads a small GGUF model
- Benchmarks for MatMul and attention
