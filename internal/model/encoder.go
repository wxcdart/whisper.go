package model

import (
	"context"
	"fmt"

	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/ml"
)

const layerNormEps = float32(1e-5)

// selfAttn holds Q/K/V/out weight and bias tensors for one self-attention layer.
type selfAttn struct {
	qW, qB    ml.Tensor // [nAudioState, nAudioState], [nAudioState]
	kW        ml.Tensor // [nAudioState, nAudioState] (no bias)
	vW, vB    ml.Tensor // [nAudioState, nAudioState], [nAudioState]
	outW, outB ml.Tensor // [nAudioState, nAudioState], [nAudioState]
}

// encoderBlock holds all weight tensors for one transformer encoder block.
type encoderBlock struct {
	attnLnW, attnLnB ml.Tensor // [nAudioState]
	attn              selfAttn
	mlpLnW, mlpLnB   ml.Tensor // [nAudioState]
	mlp0W, mlp0B     ml.Tensor // [4*nAudioState, nAudioState], [4*nAudioState]
	mlp2W, mlp2B     ml.Tensor // [nAudioState, 4*nAudioState], [nAudioState]
}

// WhisperEncoder implements the Encoder interface.
type WhisperEncoder struct {
	nMels       int
	nAudioState int
	nHead       int
	nAudioLayer int
	nAudioCtx   int

	conv1W, conv1B  ml.Tensor // [nAudioState, nMels, 3], [nAudioState]
	conv2W, conv2B  ml.Tensor // [nAudioState, nAudioState, 3], [nAudioState]
	posEmb          ml.Tensor // [nAudioCtx, nAudioState]
	blocks          []encoderBlock
	lnPostW, lnPostB ml.Tensor // [nAudioState]
}

// loadTensor loads a named tensor from f and wraps it as ml.Tensor.
func loadTensor(ctx context.Context, f *gguf.File, name string) (ml.Tensor, error) {
	data, shape, err := f.Tensor(ctx, name)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: load tensor %q: %w", name, err)
	}
	return ml.From(data, shape...), nil
}

// NewEncoder loads an encoder from an open GGUF file.
// It reads hyperparameters from metadata and pre-loads all encoder weights as float32 tensors.
func NewEncoder(f *gguf.File) (*WhisperEncoder, error) {
	ctx := context.Background()

	getMeta := func(key string) (uint32, error) {
		v, ok := f.MetaUint32(key)
		if !ok {
			return 0, fmt.Errorf("model: encoder: missing metadata %q", key)
		}
		return v, nil
	}

	nHead, err := getMeta("whisper.encoder.attention.head_count")
	if err != nil {
		return nil, err
	}
	nLayers, err := getMeta("whisper.encoder.layer_count")
	if err != nil {
		return nil, err
	}
	nAudioCtx, err := getMeta("whisper.encoder.context_length")
	if err != nil {
		return nil, err
	}
	nAudioState, err := getMeta("whisper.encoder.embedding_length")
	if err != nil {
		return nil, err
	}
	nMels, err := getMeta("whisper.audio.mel_count")
	if err != nil {
		return nil, err
	}

	e := &WhisperEncoder{
		nMels:       int(nMels),
		nAudioState: int(nAudioState),
		nHead:       int(nHead),
		nAudioLayer: int(nLayers),
		nAudioCtx:   int(nAudioCtx),
		blocks:      make([]encoderBlock, nLayers),
	}

	// Load conv stem weights.
	// GGUF stores conv1.weight as [nMels, nAudioState, 3]; Conv1D expects [outC, inC, kernel].
	t, err := loadTensor(ctx, f, "encoder.conv1.weight")
	if err != nil {
		return nil, err
	}
	e.conv1W = ml.Transpose(t, 1, 0, 2)

	if e.conv1B, err = loadTensor(ctx, f, "encoder.conv1.bias"); err != nil {
		return nil, err
	}

	t, err = loadTensor(ctx, f, "encoder.conv2.weight")
	if err != nil {
		return nil, err
	}
	e.conv2W = ml.Transpose(t, 1, 0, 2)

	if e.conv2B, err = loadTensor(ctx, f, "encoder.conv2.bias"); err != nil {
		return nil, err
	}

	if e.posEmb, err = loadTensor(ctx, f, "encoder.positional_embedding"); err != nil {
		return nil, err
	}

	// Load transformer blocks.
	for i := 0; i < int(nLayers); i++ {
		b := &e.blocks[i]
		p := fmt.Sprintf("encoder.blocks.%d", i)

		if b.attnLnW, err = loadTensor(ctx, f, p+".attn_ln.weight"); err != nil {
			return nil, err
		}
		if b.attnLnB, err = loadTensor(ctx, f, p+".attn_ln.bias"); err != nil {
			return nil, err
		}
		// Attention weights are [nAudioState, nAudioState] — used directly with MatMulTransB.
		if b.attn.qW, err = loadTensor(ctx, f, p+".attn.query.weight"); err != nil {
			return nil, err
		}
		if b.attn.qB, err = loadTensor(ctx, f, p+".attn.query.bias"); err != nil {
			return nil, err
		}
		if b.attn.kW, err = loadTensor(ctx, f, p+".attn.key.weight"); err != nil {
			return nil, err
		}
		if b.attn.vW, err = loadTensor(ctx, f, p+".attn.value.weight"); err != nil {
			return nil, err
		}
		if b.attn.vB, err = loadTensor(ctx, f, p+".attn.value.bias"); err != nil {
			return nil, err
		}
		if b.attn.outW, err = loadTensor(ctx, f, p+".attn.out.weight"); err != nil {
			return nil, err
		}
		if b.attn.outB, err = loadTensor(ctx, f, p+".attn.out.bias"); err != nil {
			return nil, err
		}
		if b.mlpLnW, err = loadTensor(ctx, f, p+".mlp_ln.weight"); err != nil {
			return nil, err
		}
		if b.mlpLnB, err = loadTensor(ctx, f, p+".mlp_ln.bias"); err != nil {
			return nil, err
		}
		// GGUF mlp.0.weight is [nAudioState, 4*nAudioState]; MatMulTransB needs [4D, D].
		t, err = loadTensor(ctx, f, p+".mlp.0.weight")
		if err != nil {
			return nil, err
		}
		b.mlp0W = ml.Transpose(t, 1, 0)

		if b.mlp0B, err = loadTensor(ctx, f, p+".mlp.0.bias"); err != nil {
			return nil, err
		}
		// GGUF mlp.2.weight is [4*nAudioState, nAudioState]; MatMulTransB needs [D, 4D].
		t, err = loadTensor(ctx, f, p+".mlp.2.weight")
		if err != nil {
			return nil, err
		}
		b.mlp2W = ml.Transpose(t, 1, 0)

		if b.mlp2B, err = loadTensor(ctx, f, p+".mlp.2.bias"); err != nil {
			return nil, err
		}
	}

	if e.lnPostW, err = loadTensor(ctx, f, "encoder.ln_post.weight"); err != nil {
		return nil, err
	}
	if e.lnPostB, err = loadTensor(ctx, f, "encoder.ln_post.bias"); err != nil {
		return nil, err
	}

	return e, nil
}

// Encode runs the encoder forward pass. mel has shape [n_mels, T] (T must be ≤ 2*n_audio_ctx).
// Returns encoder hidden states of shape [T/2, n_audio_state].
func (e *WhisperEncoder) Encode(ctx context.Context, mel ml.Tensor) (ml.Tensor, error) {
	if len(mel.Shape) != 2 || mel.Shape[0] != e.nMels {
		return ml.Tensor{}, fmt.Errorf("model: encoder: mel must be [%d, T], got %v", e.nMels, mel.Shape)
	}
	T := mel.Shape[1]
	if T > 2*e.nAudioCtx {
		return ml.Tensor{}, fmt.Errorf("model: encoder: T=%d > 2*n_audio_ctx=%d", T, 2*e.nAudioCtx)
	}

	// Conv stem: [nMels, T] → [nAudioState, T] → GELU → [nAudioState, T/2] → GELU.
	x, err := ml.Conv1D(ctx, mel, e.conv1W, e.conv1B, 1)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: encoder: conv1: %w", err)
	}
	x = ml.GELU(x)

	x, err = ml.Conv1D(ctx, x, e.conv2W, e.conv2B, 2)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: encoder: conv2: %w", err)
	}
	x = ml.GELU(x)

	// Transpose [nAudioState, T'] → [T', nAudioState] to match positional embedding layout.
	x = ml.Transpose(x, 1, 0)
	Tprime := x.Shape[0]

	// Add positional embedding (first Tprime rows of [nAudioCtx, nAudioState]).
	posSlice := ml.From(e.posEmb.Data[:Tprime*e.nAudioState], Tprime, e.nAudioState)
	x = ml.Add(x, posSlice)

	headDim := e.nAudioState / e.nHead

	// Transformer encoder blocks.
	for i := range e.blocks {
		b := &e.blocks[i]

		// Self-attention sub-layer.
		xln := ml.LayerNorm(x, b.attnLnW, b.attnLnB, layerNormEps)
		attnOut, err := e.runSelfAttn(ctx, xln, b.attn, Tprime, headDim)
		if err != nil {
			return ml.Tensor{}, fmt.Errorf("model: encoder: block %d attn: %w", i, err)
		}
		x = ml.Add(x, attnOut)

		// MLP sub-layer.
		xln = ml.LayerNorm(x, b.mlpLnW, b.mlpLnB, layerNormEps)
		mlpOut, err := e.runMLP(ctx, xln, b)
		if err != nil {
			return ml.Tensor{}, fmt.Errorf("model: encoder: block %d mlp: %w", i, err)
		}
		x = ml.Add(x, mlpOut)

		if err := ctx.Err(); err != nil {
			return ml.Tensor{}, fmt.Errorf("model: encoder: cancelled after block %d: %w", i, err)
		}
	}

	// Final layer norm.
	x = ml.LayerNorm(x, e.lnPostW, e.lnPostB, layerNormEps)
	return x, nil
}

// runSelfAttn computes multi-head self-attention for the encoder (no causal mask).
// x has shape [T, nAudioState]; returns [T, nAudioState].
func (e *WhisperEncoder) runSelfAttn(ctx context.Context, x ml.Tensor, a selfAttn, T, headDim int) (ml.Tensor, error) {
	// Q = x @ qW^T + qB
	Q, err := ml.MatMulTransB(ctx, x, a.qW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn Q: %w", err)
	}
	Q = ml.Add(Q, a.qB)

	// K = x @ kW^T (no bias)
	K, err := ml.MatMulTransB(ctx, x, a.kW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn K: %w", err)
	}

	// V = x @ vW^T + vB
	V, err := ml.MatMulTransB(ctx, x, a.vW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn V: %w", err)
	}
	V = ml.Add(V, a.vB)

	// Reshape [T, D] → [T, nHead, headDim] → [nHead, T, headDim].
	Q = ml.Transpose(Q.Reshape(T, e.nHead, headDim), 1, 0, 2)
	K = ml.Transpose(K.Reshape(T, e.nHead, headDim), 1, 0, 2)
	V = ml.Transpose(V.Reshape(T, e.nHead, headDim), 1, 0, 2)

	// Scaled dot-product attention (no causal mask for encoder).
	attnOut, _, err := ml.ScaledDotProductAttention(ctx, Q, K, V, false, false)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn sdp: %w", err)
	}

	// [nHead, T, headDim] → [T, nHead, headDim] → [T, D].
	attnOut = ml.Transpose(attnOut, 1, 0, 2).Reshape(T, e.nAudioState)

	// Output projection: attnOut @ outW^T + outB.
	out, err := ml.MatMulTransB(ctx, attnOut, a.outW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn out: %w", err)
	}
	out = ml.Add(out, a.outB)
	return out, nil
}

// runMLP computes the two-layer MLP sub-layer for one encoder block.
// x has shape [T, nAudioState]; returns [T, nAudioState].
func (e *WhisperEncoder) runMLP(ctx context.Context, x ml.Tensor, b *encoderBlock) (ml.Tensor, error) {
	// Linear 1: [T, D] → [T, 4D]
	h, err := ml.MatMulTransB(ctx, x, b.mlp0W)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("mlp linear1: %w", err)
	}
	h = ml.Add(h, b.mlp0B)
	h = ml.GELU(h)

	// Linear 2: [T, 4D] → [T, D]
	out, err := ml.MatMulTransB(ctx, h, b.mlp2W)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("mlp linear2: %w", err)
	}
	out = ml.Add(out, b.mlp2B)
	return out, nil
}
