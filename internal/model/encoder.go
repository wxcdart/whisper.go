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
	qW, qB     ml.Tensor // [nAudioState, nAudioState], [nAudioState]
	qWQ        *ml.QuantizedMatrix
	kW         ml.Tensor // [nAudioState, nAudioState] (no bias)
	kWQ        *ml.QuantizedMatrix
	vW, vB     ml.Tensor // [nAudioState, nAudioState], [nAudioState]
	vWQ        *ml.QuantizedMatrix
	outW, outB ml.Tensor // [nAudioState, nAudioState], [nAudioState]
	outWQ      *ml.QuantizedMatrix
}

// encoderBlock holds all weight tensors for one transformer encoder block.
type encoderBlock struct {
	attnLnW, attnLnB ml.Tensor // [nAudioState]
	attn             selfAttn
	mlpLnW, mlpLnB   ml.Tensor // [nAudioState]
	mlp0W, mlp0B     ml.Tensor // [4*nAudioState, nAudioState], [4*nAudioState]
	mlp0WQ           *ml.QuantizedMatrix
	mlp2W, mlp2B     ml.Tensor // [nAudioState, 4*nAudioState], [nAudioState]
	mlp2WQ           *ml.QuantizedMatrix
}

// WhisperEncoder implements the Encoder interface.
type WhisperEncoder struct {
	nMels       int
	nAudioState int
	nHead       int
	nAudioLayer int
	nAudioCtx   int

	conv1W, conv1B   ml.Tensor // [nAudioState, nMels, 3], [nAudioState]
	conv2W, conv2B   ml.Tensor // [nAudioState, nAudioState, 3], [nAudioState]
	posEmb           ml.Tensor // [nAudioCtx, nAudioState]
	blocks           []encoderBlock
	lnPostW, lnPostB ml.Tensor // [nAudioState]
}

// loadTensor loads a named tensor from f and wraps it as ml.Tensor.
func loadTensor(ctx context.Context, f gguf.FileLike, name string) (ml.Tensor, error) {
	data, shape, err := f.Tensor(ctx, name)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: load tensor %q: %w", name, err)
	}
	return ml.From(data, shape...), nil
}

// NewEncoder loads an encoder from an open GGUF file.
// It reads hyperparameters from metadata and pre-loads all encoder weights as float32 tensors.
func NewEncoder(f gguf.FileLike) (*WhisperEncoder, error) {
	ctx := context.Background()

	nHead, _ := getMetaAny(f, "whisper.encoder.attention.head_count")
	nLayers, _ := getMetaAny(f, "whisper.encoder.layer_count")
	nAudioCtx, _ := getMetaAny(f, "whisper.encoder.context_length")
	nAudioState, _ := getMetaAny(f, "whisper.encoder.embedding_length")
	nMels, _ := getMetaAny(f, "whisper.audio.mel_count", "whisper.n_mels")

	if nLayers == 0 {
		if c, ok := inferLayerCount(f, "encoder.blocks.%d.attn.query.weight"); ok {
			nLayers = uint32(c)
		}
	}

	if nAudioState == 0 {
		b, _, err := loadTensorAuto(ctx, f, "encoder.conv1.bias")
		if err != nil {
			return nil, err
		}
		if len(b.Shape) != 1 || b.Shape[0] <= 0 {
			return nil, fmt.Errorf("model: encoder: invalid conv1.bias shape %v", b.Shape)
		}
		nAudioState = uint32(b.Shape[0])
	}

	if attnShape, err := loadTensorShape(ctx, f, "encoder.blocks.0.attn.query.weight"); err == nil {
		if len(attnShape) == 2 && attnShape[0] == attnShape[1] && attnShape[0] > 0 {
			nAudioState = uint32(attnShape[0])
		}
	}

	if nAudioCtx == 0 {
		posShape, err := loadTensorShape(ctx, f, "encoder.positional_embedding")
		if err != nil {
			return nil, err
		}
		if len(posShape) != 2 {
			return nil, fmt.Errorf("model: encoder: invalid positional embedding shape %v", posShape)
		}
		if nAudioState > 0 {
			if posShape[0] == int(nAudioState) && posShape[1] != int(nAudioState) {
				nAudioCtx = uint32(posShape[1])
			} else {
				nAudioCtx = uint32(posShape[0])
			}
		} else {
			nAudioCtx = uint32(posShape[0])
			nAudioState = uint32(posShape[1])
		}
	}

	if nHead == 0 {
		if nAudioState%whisperHeadDim != 0 {
			return nil, fmt.Errorf("model: encoder: cannot infer head count from embedding length %d", nAudioState)
		}
		nHead = nAudioState / whisperHeadDim
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
	// Legacy whisper.cpp GGUF stores conv1 as [nMels, nAudioState, 3],
	// while HuggingFace-style exports often store [nAudioState, nMels, 3].
	t, resolvedName, err := loadTensorAuto(ctx, f, "encoder.conv1.weight")
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 3 {
		return nil, fmt.Errorf("model: encoder: invalid conv1.weight shape %v", t.Shape)
	}
	if t.Shape[0] == int(nAudioState) && t.Shape[1] > 0 && t.Shape[2] == 3 {
		e.conv1W = t
		if nMels == 0 {
			nMels = uint32(t.Shape[1])
			e.nMels = int(nMels)
		}
	} else if t.Shape[1] == int(nAudioState) && t.Shape[0] > 0 && t.Shape[2] == 3 {
		e.conv1W = ml.Transpose(t, 1, 0, 2)
		if nMels == 0 {
			nMels = uint32(t.Shape[0])
			e.nMels = int(nMels)
		}
	} else if t.Shape[0] == 3 && t.Shape[2] == int(nAudioState) && t.Shape[1] > 0 {
		e.conv1W = ml.Transpose(t, 2, 1, 0)
		if nMels == 0 {
			nMels = uint32(t.Shape[1])
			e.nMels = int(nMels)
		}
	} else {
		return nil, fmt.Errorf("model: encoder: unsupported conv1.weight shape %v (from %q)", t.Shape, resolvedName)
	}

	if e.conv1B, _, err = loadTensorAuto(ctx, f, "encoder.conv1.bias"); err != nil {
		return nil, err
	}

	t, _, err = loadTensorAuto(ctx, f, "encoder.conv2.weight")
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 3 {
		return nil, fmt.Errorf("model: encoder: invalid conv2.weight shape %v", t.Shape)
	}
	// conv2 in/out channels are both nAudioState in Whisper.
	if t.Shape[0] == int(nAudioState) && t.Shape[1] == int(nAudioState) && t.Shape[2] == 3 {
		e.conv2W = t
	} else if t.Shape[0] == 3 && t.Shape[1] == int(nAudioState) && t.Shape[2] == int(nAudioState) {
		e.conv2W = ml.Transpose(t, 2, 1, 0)
	} else {
		e.conv2W = ml.Transpose(t, 1, 0, 2)
	}

	if e.conv2B, _, err = loadTensorAuto(ctx, f, "encoder.conv2.bias"); err != nil {
		return nil, err
	}

	if e.posEmb, _, err = loadTensorAuto(ctx, f, "encoder.positional_embedding"); err != nil {
		return nil, err
	}
	if len(e.posEmb.Shape) != 2 {
		return nil, fmt.Errorf("model: encoder: invalid positional embedding shape %v", e.posEmb.Shape)
	}
	if e.posEmb.Shape[0] == e.nAudioState && e.posEmb.Shape[1] == e.nAudioCtx {
		e.posEmb = ml.Transpose(e.posEmb, 1, 0)
	}

	// Load transformer blocks.
	for i := 0; i < int(nLayers); i++ {
		b := &e.blocks[i]
		p := fmt.Sprintf("encoder.blocks.%d", i)

		if b.attnLnW, _, err = loadTensorAuto(ctx, f, p+".attn_ln.weight"); err != nil {
			return nil, err
		}
		if b.attnLnB, _, err = loadTensorAuto(ctx, f, p+".attn_ln.bias"); err != nil {
			return nil, err
		}
		// Attention weights are [nAudioState, nAudioState] — used directly with MatMulTransB.
		if b.attn.qW, b.attn.qWQ, _, err = loadMatWeightAuto(ctx, f, p+".attn.query.weight", int(nAudioState), int(nAudioState)); err != nil {
			return nil, err
		}
		if b.attn.qB, _, err = loadTensorAuto(ctx, f, p+".attn.query.bias"); err != nil {
			return nil, err
		}
		if b.attn.kW, b.attn.kWQ, _, err = loadMatWeightAuto(ctx, f, p+".attn.key.weight", int(nAudioState), int(nAudioState)); err != nil {
			return nil, err
		}
		if b.attn.vW, b.attn.vWQ, _, err = loadMatWeightAuto(ctx, f, p+".attn.value.weight", int(nAudioState), int(nAudioState)); err != nil {
			return nil, err
		}
		if b.attn.vB, _, err = loadTensorAuto(ctx, f, p+".attn.value.bias"); err != nil {
			return nil, err
		}
		if b.attn.outW, b.attn.outWQ, _, err = loadMatWeightAuto(ctx, f, p+".attn.out.weight", int(nAudioState), int(nAudioState)); err != nil {
			return nil, err
		}
		if b.attn.outB, _, err = loadTensorAuto(ctx, f, p+".attn.out.bias"); err != nil {
			return nil, err
		}
		if b.mlpLnW, _, err = loadTensorAuto(ctx, f, p+".mlp_ln.weight"); err != nil {
			return nil, err
		}
		if b.mlpLnB, _, err = loadTensorAuto(ctx, f, p+".mlp_ln.bias"); err != nil {
			return nil, err
		}
		b.mlp0W, b.mlp0WQ, _, err = loadMatWeightAuto(ctx, f, p+".mlp.0.weight", 4*int(nAudioState), int(nAudioState))
		if err != nil {
			return nil, err
		}

		if b.mlp0B, _, err = loadTensorAuto(ctx, f, p+".mlp.0.bias"); err != nil {
			return nil, err
		}
		b.mlp2W, b.mlp2WQ, _, err = loadMatWeightAuto(ctx, f, p+".mlp.2.weight", int(nAudioState), 4*int(nAudioState))
		if err != nil {
			return nil, err
		}

		if b.mlp2B, _, err = loadTensorAuto(ctx, f, p+".mlp.2.bias"); err != nil {
			return nil, err
		}
	}

	if e.lnPostW, _, err = loadTensorAuto(ctx, f, "encoder.ln_post.weight"); err != nil {
		return nil, err
	}
	if e.lnPostB, _, err = loadTensorAuto(ctx, f, "encoder.ln_post.bias"); err != nil {
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
	if T == 2*e.nAudioCtx+1 {
		// Some pipelines produce one extra STFT frame at chunk boundaries.
		mel = ml.From(mel.Data[:e.nMels*(2*e.nAudioCtx)], e.nMels, 2*e.nAudioCtx)
		T = mel.Shape[1]
	}
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
	Q, err := matMulTransBMaybeQuant(ctx, x, a.qW, a.qWQ)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn Q: %w", err)
	}
	Q = ml.Add(Q, a.qB)

	// K = x @ kW^T (no bias)
	K, err := matMulTransBMaybeQuant(ctx, x, a.kW, a.kWQ)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("attn K: %w", err)
	}

	// V = x @ vW^T + vB
	V, err := matMulTransBMaybeQuant(ctx, x, a.vW, a.vWQ)
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
	out, err := matMulTransBMaybeQuant(ctx, attnOut, a.outW, a.outWQ)
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
	h, err := matMulTransBMaybeQuant(ctx, x, b.mlp0W, b.mlp0WQ)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("mlp linear1: %w", err)
	}
	h = ml.Add(h, b.mlp0B)
	h = ml.GELU(h)

	// Linear 2: [T, 4D] → [T, D]
	out, err := matMulTransBMaybeQuant(ctx, h, b.mlp2W, b.mlp2WQ)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("mlp linear2: %w", err)
	}
	out = ml.Add(out, b.mlp2B)
	return out, nil
}
