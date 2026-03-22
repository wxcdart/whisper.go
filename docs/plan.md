GGML Go Port — Plan

Roadmap (phases)
- Phase 0: Scoping & benchmarks (2-3 days)
- Phase 1: Tensor core + MatMul prototype (7-10 days)
- Phase 2: Quant/dequant integration + tests (3-5 days)
- Phase 3: Core ops implementation (7-10 days)
- Phase 4: Attention + encoder/decoder glue (7-12 days)
- Phase 5: Performance tuning (tiling, BLAS, optional SIMD) (10-20 days)
- Phase 6: Integration tests, CI, docs (3-5 days)

Milestones
- M1: MatMul prototype + unit tests
- M2: Dequant parity tests for Q4_K
- M3: Run a full whisper encoder/decoder using small model
- M4: Achieve 70-85% of ggml baseline on representative hardware (optimization phase)

Acceptance criteria (MVP)
- All core ops pass unit tests (tolerance defined)
- Able to load a GGUF model and run transcription for a short audio file
- Benchmarks included and CI job runs on PRs
