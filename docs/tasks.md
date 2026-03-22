GGML Go Port — Tasks & Worktrees

Tasks (short term)
- Create `internal/ml` scaffold and `Tensor` type
- Implement `MatMul` with gonum BLAS path
- Integrate existing `internal/gguf` dequant functions
- Add unit tests and small GGUF fixture tests
- Add CI job and benchmarks

Using Git worktrees to parallelize development
1. Create a main integration worktree (keep `main` branch clean):

   git worktree add ../whisper-main main

2. Create per-feature worktrees (examples):

   git worktree add ../whisper-tensor -b feat/tensor main
   git worktree add ../whisper-quant -b feat/quant main
   git worktree add ../whisper-attn  -b feat/attention main

3. Workflow suggestions
- Work in feature worktrees; commit and push branches regularly.
- Open a single PR per feature branch; use CI to run unit tests and benchmarks.
- Rebase feature branches onto `main` before merge to reduce conflicts.

4. Cleanup
- Remove a worktree after merging:

   git worktree remove ../whisper-tensor
   git branch -D feat/tensor

Notes
- Keep worktrees on the same filesystem for speed; each worktree shares the same git object db.
- Use `go test ./...` in each worktree to validate changes locally before pushing.
