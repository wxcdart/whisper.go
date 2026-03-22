package model

import (
	"context"
	"sync"
	"time"

	"github.com/whispergo/whisper.go/internal/ml"
)

// OpTimingStats contains aggregate timing data for one backend operation.
type OpTimingStats struct {
	Calls int64
	Total time.Duration
}

// TimedBackend wraps a ComputeBackend and records per-operation call counts and latency totals.
type TimedBackend struct {
	base ComputeBackend

	mu    sync.Mutex
	stats map[string]OpTimingStats
}

// NewTimedBackend returns a backend wrapper that records timing statistics.
// If base is nil, the default ml backend is used.
func NewTimedBackend(base ComputeBackend) *TimedBackend {
	if base == nil {
		base = NewMLComputeBackend()
	}
	return &TimedBackend{
		base:  base,
		stats: make(map[string]OpTimingStats),
	}
}

// Snapshot returns a copy of all currently recorded operation stats.
func (t *TimedBackend) Snapshot() map[string]OpTimingStats {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make(map[string]OpTimingStats, len(t.stats))
	for k, v := range t.stats {
		out[k] = v
	}
	return out
}

// Reset clears all recorded timing stats.
func (t *TimedBackend) Reset() {
	t.mu.Lock()
	clear(t.stats)
	t.mu.Unlock()
}

func (t *TimedBackend) record(op string, started time.Time) {
	t.mu.Lock()
	s := t.stats[op]
	s.Calls++
	s.Total += time.Since(started)
	t.stats[op] = s
	t.mu.Unlock()
}

func (t *TimedBackend) Conv1D(ctx context.Context, x, w, b ml.Tensor, stride int) (out ml.Tensor, err error) {
	started := time.Now()
	defer t.record("Conv1D", started)
	return t.base.Conv1D(ctx, x, w, b, stride)
}

func (t *TimedBackend) GELU(x ml.Tensor) (out ml.Tensor) {
	started := time.Now()
	defer t.record("GELU", started)
	return t.base.GELU(x)
}

func (t *TimedBackend) GELUInPlace(x ml.Tensor) {
	started := time.Now()
	defer t.record("GELUInPlace", started)
	t.base.GELUInPlace(x)
}

func (t *TimedBackend) Add(a, b ml.Tensor) (out ml.Tensor) {
	started := time.Now()
	defer t.record("Add", started)
	return t.base.Add(a, b)
}

func (t *TimedBackend) LayerNorm(x, weight, bias ml.Tensor, eps float32) (out ml.Tensor) {
	started := time.Now()
	defer t.record("LayerNorm", started)
	return t.base.LayerNorm(x, weight, bias, eps)
}

func (t *TimedBackend) LayerNormInto(dst, x, weight, bias ml.Tensor, eps float32) (err error) {
	started := time.Now()
	defer t.record("LayerNormInto", started)
	return t.base.LayerNormInto(dst, x, weight, bias, eps)
}

func (t *TimedBackend) Transpose(x ml.Tensor, axes ...int) (out ml.Tensor) {
	started := time.Now()
	defer t.record("Transpose", started)
	return t.base.Transpose(x, axes...)
}

func (t *TimedBackend) ScaledDotProductAttention(ctx context.Context, q, k, v ml.Tensor, causal, returnScores bool) (out, scores ml.Tensor, err error) {
	started := time.Now()
	defer t.record("ScaledDotProductAttention", started)
	return t.base.ScaledDotProductAttention(ctx, q, k, v, causal, returnScores)
}

func (t *TimedBackend) ScaledDotProductAttentionInto(ctx context.Context, q, k, v ml.Tensor, causal bool, out, scratch, scores ml.Tensor) (err error) {
	started := time.Now()
	defer t.record("ScaledDotProductAttentionInto", started)
	return t.base.ScaledDotProductAttentionInto(ctx, q, k, v, causal, out, scratch, scores)
}

func (t *TimedBackend) MatMulTransB(ctx context.Context, a, b ml.Tensor) (out ml.Tensor, err error) {
	started := time.Now()
	defer t.record("MatMulTransB", started)
	return t.base.MatMulTransB(ctx, a, b)
}

func (t *TimedBackend) MatMulTransBInto(ctx context.Context, a, b, out ml.Tensor) (err error) {
	started := time.Now()
	defer t.record("MatMulTransBInto", started)
	return t.base.MatMulTransBInto(ctx, a, b, out)
}

func (t *TimedBackend) MatMulQuantTransBInto(ctx context.Context, a ml.Tensor, b ml.QuantizedMatrix, out ml.Tensor) (err error) {
	started := time.Now()
	defer t.record("MatMulQuantTransBInto", started)
	return t.base.MatMulQuantTransBInto(ctx, a, b, out)
}

func (t *TimedBackend) ShouldUseQuantMatMul(m, k, n int, qtype uint32) bool {
	started := time.Now()
	defer t.record("ShouldUseQuantMatMul", started)
	return t.base.ShouldUseQuantMatMul(m, k, n, qtype)
}
