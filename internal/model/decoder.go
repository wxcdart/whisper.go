package model

import (
	"context"
	"fmt"
	"math"
	"math/rand"

	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/ml"
)

// decoderCrossAttn holds Q/K/V/out weights for cross-attention to encoder output.
type decoderCrossAttn struct {
	qW, qB    ml.Tensor // [nTextState, nTextState], [nTextState]
	kW, vW    ml.Tensor // [nTextState, nTextState] (no bias)
	vB        ml.Tensor // [nTextState]
	outW, outB ml.Tensor // [nTextState, nTextState], [nTextState]
}

// decoderSelfAttn holds Q/K/V/out weights for causal self-attention (with KV cache).
type decoderSelfAttn struct {
	qW, qB    ml.Tensor // [nTextState, nTextState], [nTextState]
	kW        ml.Tensor // [nTextState, nTextState] (no bias)
	vW, vB    ml.Tensor // [nTextState, nTextState], [nTextState]
	outW, outB ml.Tensor // [nTextState, nTextState], [nTextState]
}

// decoderBlock holds all weight tensors for one transformer decoder block.
type decoderBlock struct {
	// Self-attention
	sAttnLnW, sAttnLnB ml.Tensor // [nTextState]
	selfAttn            decoderSelfAttn

	// Cross-attention
	cAttnLnW, cAttnLnB ml.Tensor // [nTextState]
	crossAttn           decoderCrossAttn

	// MLP
	mlpLnW, mlpLnB ml.Tensor // [nTextState]
	mlp0W, mlp0B   ml.Tensor // [4*nTextState, nTextState], [4*nTextState]
	mlp2W, mlp2B   ml.Tensor // [nTextState, 4*nTextState], [nTextState]
}

// WhisperDecoder implements the Decoder interface.
type WhisperDecoder struct {
	nVocab      int
	nTextState  int
	nHead       int
	nTextLayer  int
	nTextCtx    int

	// Embeddings
	tokenEmb ml.Tensor // [nVocab, nTextState]
	posEmb   ml.Tensor // [nTextCtx, nTextState]

	// Transformer blocks
	blocks []decoderBlock

	// Final layer norm
	lnW, lnB ml.Tensor // [nTextState]

	// Special token IDs
	eotToken int32
}

// NewDecoder loads a decoder from an open GGUF file.
func NewDecoder(f *gguf.File) (*WhisperDecoder, error) {
	ctx := context.Background()
	var err error

	nHead, _ := getMetaAny(f, "whisper.decoder.attention.head_count")
	nLayers, _ := getMetaAny(f, "whisper.decoder.layer_count")
	nTextCtx, _ := getMetaAny(f, "whisper.decoder.context_length")
	nTextState, _ := getMetaAny(f, "whisper.decoder.embedding_length")
	nVocab, _ := getMetaAny(f, "whisper.vocab.size", "whisper.n_vocab")

	if nLayers == 0 {
		if c, ok := inferLayerCount(f, "decoder.blocks.%d.self_attn.query.weight"); ok {
			nLayers = uint32(c)
		}
	}

	if attnShape, err := loadTensorShape(ctx, f, "decoder.blocks.0.self_attn.query.weight"); err == nil {
		if len(attnShape) == 2 && attnShape[0] == attnShape[1] && attnShape[0] > 0 {
			nTextState = uint32(attnShape[0])
		}
	}

	if nTextCtx == 0 || nTextState == 0 {
		posShape, err := loadTensorShape(ctx, f, "decoder.positional_embedding")
		if err != nil {
			return nil, err
		}
		if len(posShape) != 2 {
			return nil, fmt.Errorf("model: decoder: invalid positional embedding shape %v", posShape)
		}
		if nTextState > 0 {
			if posShape[0] == int(nTextState) && posShape[1] != int(nTextState) {
				nTextCtx = uint32(posShape[1])
			} else if posShape[1] == int(nTextState) {
				nTextCtx = uint32(posShape[0])
			}
		} else {
			nTextCtx = uint32(posShape[0])
			nTextState = uint32(posShape[1])
		}
	}

	if nVocab == 0 {
		tokShape, err := loadTensorShape(ctx, f, "decoder.token_embedding.weight")
		if err != nil {
			return nil, err
		}
		if len(tokShape) != 2 {
			return nil, fmt.Errorf("model: decoder: invalid token embedding shape %v", tokShape)
		}
		if nTextState > 0 {
			if tokShape[1] == int(nTextState) {
				nVocab = uint32(tokShape[0])
			} else if tokShape[0] == int(nTextState) {
				nVocab = uint32(tokShape[1])
			}
		} else {
			nVocab = uint32(tokShape[0])
			nTextState = uint32(tokShape[1])
		}
	}

	if nHead == 0 {
		if nTextState%whisperHeadDim != 0 {
			return nil, fmt.Errorf("model: decoder: cannot infer head count from embedding length %d", nTextState)
		}
		nHead = nTextState / whisperHeadDim
	}

	d := &WhisperDecoder{
		nVocab:     int(nVocab),
		nTextState: int(nTextState),
		nHead:      int(nHead),
		nTextLayer: int(nLayers),
		nTextCtx:   int(nTextCtx),
		blocks:     make([]decoderBlock, nLayers),
		eotToken:   50256, // Default Whisper EOT token ID
	}

	// Load token embedding [nVocab, nTextState].
	if d.tokenEmb, _, err = loadTensorAuto(ctx, f, "decoder.token_embedding.weight"); err != nil {
		return nil, err
	}
	if len(d.tokenEmb.Shape) != 2 {
		return nil, fmt.Errorf("model: decoder: invalid token embedding shape %v", d.tokenEmb.Shape)
	}
	if d.tokenEmb.Shape[0] == d.nTextState && d.tokenEmb.Shape[1] == d.nVocab {
		d.tokenEmb = ml.Transpose(d.tokenEmb, 1, 0)
	}

	// Load positional embedding [nTextCtx, nTextState].
	if d.posEmb, _, err = loadTensorAuto(ctx, f, "decoder.positional_embedding"); err != nil {
		return nil, err
	}
	if len(d.posEmb.Shape) != 2 {
		return nil, fmt.Errorf("model: decoder: invalid positional embedding shape %v", d.posEmb.Shape)
	}
	if d.posEmb.Shape[0] == d.nTextState && d.posEmb.Shape[1] == d.nTextCtx {
		d.posEmb = ml.Transpose(d.posEmb, 1, 0)
	}

	// Load transformer blocks.
	for i := 0; i < int(nLayers); i++ {
		b := &d.blocks[i]
		p := fmt.Sprintf("decoder.blocks.%d", i)

		// Self-attention layer norm.
		if b.sAttnLnW, _, err = loadTensorAuto(ctx, f, p+".self_attn_ln.weight"); err != nil {
			return nil, err
		}
		if b.sAttnLnB, _, err = loadTensorAuto(ctx, f, p+".self_attn_ln.bias"); err != nil {
			return nil, err
		}

		// Self-attention weights.
		if b.selfAttn.qW, _, err = loadTensorAuto(ctx, f, p+".self_attn.query.weight"); err != nil {
			return nil, err
		}
		if b.selfAttn.qB, _, err = loadTensorAuto(ctx, f, p+".self_attn.query.bias"); err != nil {
			return nil, err
		}
		if b.selfAttn.kW, _, err = loadTensorAuto(ctx, f, p+".self_attn.key.weight"); err != nil {
			return nil, err
		}
		if b.selfAttn.vW, _, err = loadTensorAuto(ctx, f, p+".self_attn.value.weight"); err != nil {
			return nil, err
		}
		if b.selfAttn.vB, _, err = loadTensorAuto(ctx, f, p+".self_attn.value.bias"); err != nil {
			return nil, err
		}
		if b.selfAttn.outW, _, err = loadTensorAuto(ctx, f, p+".self_attn.out.weight"); err != nil {
			return nil, err
		}
		if b.selfAttn.outB, _, err = loadTensorAuto(ctx, f, p+".self_attn.out.bias"); err != nil {
			return nil, err
		}

		// Cross-attention layer norm.
		if b.cAttnLnW, _, err = loadTensorAuto(ctx, f, p+".cross_attn_ln.weight"); err != nil {
			return nil, err
		}
		if b.cAttnLnB, _, err = loadTensorAuto(ctx, f, p+".cross_attn_ln.bias"); err != nil {
			return nil, err
		}

		// Cross-attention weights.
		if b.crossAttn.qW, _, err = loadTensorAuto(ctx, f, p+".cross_attn.query.weight"); err != nil {
			return nil, err
		}
		if b.crossAttn.qB, _, err = loadTensorAuto(ctx, f, p+".cross_attn.query.bias"); err != nil {
			return nil, err
		}
		if b.crossAttn.kW, _, err = loadTensorAuto(ctx, f, p+".cross_attn.key.weight"); err != nil {
			return nil, err
		}
		if b.crossAttn.vW, _, err = loadTensorAuto(ctx, f, p+".cross_attn.value.weight"); err != nil {
			return nil, err
		}
		if b.crossAttn.vB, _, err = loadTensorAuto(ctx, f, p+".cross_attn.value.bias"); err != nil {
			return nil, err
		}
		if b.crossAttn.outW, _, err = loadTensorAuto(ctx, f, p+".cross_attn.out.weight"); err != nil {
			return nil, err
		}
		if b.crossAttn.outB, _, err = loadTensorAuto(ctx, f, p+".cross_attn.out.bias"); err != nil {
			return nil, err
		}

		// MLP layer norm.
		if b.mlpLnW, _, err = loadTensorAuto(ctx, f, p+".mlp_ln.weight"); err != nil {
			return nil, err
		}
		if b.mlpLnB, _, err = loadTensorAuto(ctx, f, p+".mlp_ln.bias"); err != nil {
			return nil, err
		}

		// Desired layout for MatMulTransB is [4D, D].
		t, _, err := loadTensorAuto(ctx, f, p+".mlp.0.weight")
		if err != nil {
			return nil, err
		}
		if len(t.Shape) != 2 {
			return nil, fmt.Errorf("model: decoder: invalid %s.mlp.0.weight shape %v", p, t.Shape)
		}
		if t.Shape[0] == 4*int(nTextState) && t.Shape[1] == int(nTextState) {
			b.mlp0W = t
		} else if t.Shape[0] == int(nTextState) && t.Shape[1] == 4*int(nTextState) {
			b.mlp0W = ml.Transpose(t, 1, 0)
		} else {
			return nil, fmt.Errorf("model: decoder: unsupported %s.mlp.0.weight shape %v", p, t.Shape)
		}
		if b.mlp0B, _, err = loadTensorAuto(ctx, f, p+".mlp.0.bias"); err != nil {
			return nil, err
		}

		// Desired layout for MatMulTransB is [D, 4D].
		t, _, err = loadTensorAuto(ctx, f, p+".mlp.2.weight")
		if err != nil {
			return nil, err
		}
		if len(t.Shape) != 2 {
			return nil, fmt.Errorf("model: decoder: invalid %s.mlp.2.weight shape %v", p, t.Shape)
		}
		if t.Shape[0] == int(nTextState) && t.Shape[1] == 4*int(nTextState) {
			b.mlp2W = t
		} else if t.Shape[0] == 4*int(nTextState) && t.Shape[1] == int(nTextState) {
			b.mlp2W = ml.Transpose(t, 1, 0)
		} else {
			return nil, fmt.Errorf("model: decoder: unsupported %s.mlp.2.weight shape %v", p, t.Shape)
		}
		if b.mlp2B, _, err = loadTensorAuto(ctx, f, p+".mlp.2.bias"); err != nil {
			return nil, err
		}
	}

	// Load final layer norm.
	if d.lnW, _, err = loadTensorAuto(ctx, f, "decoder.ln.weight"); err != nil {
		return nil, err
	}
	if d.lnB, _, err = loadTensorAuto(ctx, f, "decoder.ln.bias"); err != nil {
		return nil, err
	}

	return d, nil
}

// decoderState holds the running state during decoding.
type decoderState struct {
	tokens      []int32      // Current token sequence
	logits      []float32    // Latest logits [nVocab]
	kvCache     ml.Tensor    // [nLayer, 2, seqLen, nTextState]
	cachedPos   int          // Number of positions cached
}

// Decode generates token sequences using greedy or beam search.
func (d *WhisperDecoder) Decode(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error) {
	if len(encoderOut.Shape) != 2 || encoderOut.Shape[1] != d.nTextState {
		return nil, fmt.Errorf("model: decoder: encoder output must be [T, %d], got %v", d.nTextState, encoderOut.Shape)
	}

	if params.BeamSize > 1 {
		return d.decodeBeamSearch(ctx, encoderOut, params)
	}
	return d.decodeGreedy(ctx, encoderOut, params)
}

// decodeGreedy uses greedy sampling to generate tokens.
func (d *WhisperDecoder) decodeGreedy(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error) {
	state := &decoderState{
		tokens:    make([]int32, len(params.Prompt)),
		kvCache:   ml.New(d.nTextLayer, 2, d.nTextCtx, d.nTextState),
		cachedPos: 0,
	}
	copy(state.tokens, params.Prompt)

	temperature := params.Temperature
	if temperature == 0 {
		temperature = 1.0
	}

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = d.nTextCtx
	}
	if capFromCtx := d.nTextCtx - len(params.Prompt); capFromCtx < maxTokens {
		maxTokens = capFromCtx
	}
	if maxTokens < 0 {
		maxTokens = 0
	}

	for len(state.tokens) < len(params.Prompt)+maxTokens {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("model: decode: cancelled: %w", err)
		}

		// Compute logits for the next token.
		logits, err := d.forward(ctx, state, encoderOut)
		if err != nil {
			return nil, fmt.Errorf("model: decode: forward: %w", err)
		}

		// Sample next token using greedy with temperature and fallback.
		nextToken, err := d.sampleGreedy(logits, temperature, params)
		if err != nil {
			return nil, fmt.Errorf("model: decode: sample: %w", err)
		}

		state.tokens = append(state.tokens, nextToken)

		// Stop if we hit EOT.
		if nextToken == d.eotToken {
			break
		}
	}

	// Convert token sequence to segments.
	return d.tokensToSegments(state.tokens, params.Prompt)
}

// decodeBeamSearch uses beam search to generate tokens.
func (d *WhisperDecoder) decodeBeamSearch(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error) {
	beamSize := params.BeamSize
	if beamSize <= 0 {
		beamSize = 1
	}

	// Initialize beam hypotheses.
	beams := make([]*beam, beamSize)
	for i := 0; i < beamSize; i++ {
		beams[i] = &beam{
			tokens: make([]int32, len(params.Prompt)),
			score:  0,
		}
		copy(beams[i].tokens, params.Prompt)
	}

	temperature := params.Temperature
	if temperature == 0 {
		temperature = 1.0
	}

	maxTokens := params.MaxTokens
	if maxTokens <= 0 {
		maxTokens = d.nTextCtx
	}
	if capFromCtx := d.nTextCtx - len(params.Prompt); capFromCtx < maxTokens {
		maxTokens = capFromCtx
	}
	if maxTokens < 0 {
		maxTokens = 0
	}

	for step := 0; step < maxTokens; step++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("model: decode: beam search cancelled: %w", err)
		}

		// Compute logits for all beam hypotheses.
		allCandidates := []*beamCandidate{}

		for _, hyp := range beams {
			if len(hyp.tokens) > 0 && hyp.tokens[len(hyp.tokens)-1] == d.eotToken {
				// This hypothesis has reached EOT, keep it unchanged.
				allCandidates = append(allCandidates, &beamCandidate{
					hyp:    hyp,
					token:  d.eotToken,
					logprob: 0,
				})
				continue
			}

			state := &decoderState{
				tokens:    hyp.tokens,
				kvCache:   ml.New(d.nTextLayer, 2, d.nTextCtx, d.nTextState),
				cachedPos: 0,
			}

			logits, err := d.forward(ctx, state, encoderOut)
			if err != nil {
				return nil, fmt.Errorf("model: decode: beam search forward: %w", err)
			}

			// Get top-K candidates from this hypothesis.
			topK := 5 // Heuristic: keep top 5 candidates per hypothesis.
			candidates := d.topKTokens(logits, topK, temperature)

			for _, candidate := range candidates {
				newHyp := &beam{
					tokens: make([]int32, len(hyp.tokens)+1),
					score:  hyp.score + candidate.logprob,
				}
				copy(newHyp.tokens, hyp.tokens)
				newHyp.tokens[len(hyp.tokens)] = candidate.token

				// Normalize score by sequence length (length penalty).
				normScore := newHyp.score / float32(math.Pow(float64(len(newHyp.tokens)), 0.6))

				allCandidates = append(allCandidates, &beamCandidate{
					hyp:       newHyp,
					token:     candidate.token,
					logprob:   candidate.logprob,
					normScore: normScore,
				})
			}
		}

		// Sort by normalized score and keep top-K.
		bestCandidates := sortAndPruneBeams(allCandidates, beamSize)

		// Update beams with top candidates.
		beams = make([]*beam, len(bestCandidates))
		for i, cand := range bestCandidates {
			beams[i] = cand.hyp
		}

		// Check if all hypotheses have reached EOT.
		allEOT := true
		for _, hyp := range beams {
			if len(hyp.tokens) == 0 || hyp.tokens[len(hyp.tokens)-1] != d.eotToken {
				allEOT = false
				break
			}
		}
		if allEOT {
			break
		}
	}

	// Return segments from the best beam (first hypothesis).
	if len(beams) == 0 {
		return []Segment{}, nil
	}
	return d.tokensToSegments(beams[0].tokens, params.Prompt)
}

// beam represents a hypothesis in beam search.
type beam struct {
	tokens []int32
	score  float32
}

// beamCandidate represents a candidate token for beam expansion.
type beamCandidate struct {
	hyp       *beam
	token     int32
	logprob   float32
	normScore float32
}

// forward runs the decoder forward pass for the next token.
func (d *WhisperDecoder) forward(ctx context.Context, state *decoderState, encoderOut ml.Tensor) ([]float32, error) {
	if len(state.tokens) == 0 {
		return nil, fmt.Errorf("model: decode: forward: no tokens in state")
	}

	nextPos := len(state.tokens) - 1
	if nextPos >= d.nTextCtx {
		return nil, fmt.Errorf("model: decode: forward: position %d >= context length %d", nextPos, d.nTextCtx)
	}

	// Get token embedding for the last token.
	tokenID := state.tokens[nextPos]
	if tokenID < 0 || int(tokenID) >= d.nVocab {
		return nil, fmt.Errorf("model: decode: forward: invalid token ID %d", tokenID)
	}

	startIdx := int(tokenID) * d.nTextState
	endIdx := startIdx + d.nTextState
	if endIdx > len(d.tokenEmb.Data) {
		return nil, fmt.Errorf("model: decode: forward: token embedding out of bounds")
	}

	// x = token_embedding + positional_embedding [nTextState]
	x := ml.New(d.nTextState)
	copy(x.Data, d.tokenEmb.Data[startIdx:endIdx])

	posStartIdx := nextPos * d.nTextState
	for i := 0; i < d.nTextState; i++ {
		x.Data[i] += d.posEmb.Data[posStartIdx+i]
	}

	// Reshape x to [1, nTextState] for batch processing.
	x = ml.New(1, d.nTextState)
	copy(x.Data, x.Data[:d.nTextState])

	headDim := d.nTextState / d.nHead

	// Pass through decoder blocks.
	for i := range d.blocks {
		b := &d.blocks[i]

		// Self-attention with KV cache.
		xln := ml.LayerNorm(x, b.sAttnLnW, b.sAttnLnB, layerNormEps)
		attnOut, err := d.runDecoderSelfAttn(ctx, xln, b.selfAttn, nextPos, state.kvCache, i, headDim)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d self_attn: %w", i, err)
		}
		x = ml.Add(x, attnOut)

		// Cross-attention to encoder output.
		xln = ml.LayerNorm(x, b.cAttnLnW, b.cAttnLnB, layerNormEps)
		crossOut, err := d.runDecoderCrossAttn(ctx, xln, encoderOut, b.crossAttn, headDim)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d cross_attn: %w", i, err)
		}
		x = ml.Add(x, crossOut)

		// MLP sub-layer.
		xln = ml.LayerNorm(x, b.mlpLnW, b.mlpLnB, layerNormEps)
		mlpOut, err := d.runMLP(ctx, xln, b)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d mlp: %w", i, err)
		}
		x = ml.Add(x, mlpOut)

		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("model: decode: cancelled after block %d: %w", i, err)
		}
	}

	// Final layer norm.
	x = ml.LayerNorm(x, d.lnW, d.lnB, layerNormEps)

	// Project to vocabulary: x @ tokenEmb^T → [1, nVocab]
	logits, err := ml.MatMulTransB(ctx, x, d.tokenEmb)
	if err != nil {
		return nil, fmt.Errorf("model: decode: forward: logits: %w", err)
	}

	if len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] != d.nVocab {
		return nil, fmt.Errorf("model: decode: forward: logits shape mismatch: %v", logits.Shape)
	}

	return logits.Data[:d.nVocab], nil
}

// runDecoderSelfAttn runs causal self-attention with KV cache.
func (d *WhisperDecoder) runDecoderSelfAttn(ctx context.Context, x ml.Tensor, attn decoderSelfAttn, pos int, kvCache ml.Tensor, layerIdx, headDim int) (ml.Tensor, error) {
	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: x shape mismatch")
	}

	// Compute Q: x @ qW^T + qB → [1, nTextState]
	q, err := ml.MatMulTransB(ctx, x, attn.qW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: q matmul: %w", err)
	}
	q = ml.Add(q, attn.qB)

	// Compute K: x @ kW^T → [1, nTextState]
	k, err := ml.MatMulTransB(ctx, x, attn.kW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: k matmul: %w", err)
	}

	// Compute V: x @ vW^T + vB → [1, nTextState]
	v, err := ml.MatMulTransB(ctx, x, attn.vW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: v matmul: %w", err)
	}
	v = ml.Add(v, attn.vB)

	// Update KV cache with new K and V.
	// kvCache shape: [nLayer, 2, nTextCtx, nTextState]
	// Store k and v at position pos.
	nTextCtx := d.nTextCtx
	nTextState := d.nTextState
	kCacheOff := layerIdx*2*nTextCtx*nTextState + 0*nTextCtx*nTextState + pos*nTextState
	vCacheOff := layerIdx*2*nTextCtx*nTextState + 1*nTextCtx*nTextState + pos*nTextState
	copy(kvCache.Data[kCacheOff:kCacheOff+nTextState], k.Data)
	copy(kvCache.Data[vCacheOff:vCacheOff+nTextState], v.Data)

	// Retrieve accumulated K and V up to pos+1.
	// We need [nHead, pos+1, headDim] for both.
	seqLen := pos + 1
	kFull := ml.New(d.nHead, seqLen, headDim)
	vFull := ml.New(d.nHead, seqLen, headDim)

	kLayerOff := layerIdx * 2 * nTextCtx * nTextState
	vLayerOff := layerIdx*2*nTextCtx*nTextState + 1*nTextCtx*nTextState

	for h := 0; h < d.nHead; h++ {
		for s := 0; s < seqLen; s++ {
			srcKOff := kLayerOff + s*nTextState + h*headDim
			dstKOff := h*seqLen*headDim + s*headDim
			copy(kFull.Data[dstKOff:dstKOff+headDim], kvCache.Data[srcKOff:srcKOff+headDim])

			srcVOff := vLayerOff + s*nTextState + h*headDim
			dstVOff := h*seqLen*headDim + s*headDim
			copy(vFull.Data[dstVOff:dstVOff+headDim], kvCache.Data[srcVOff:srcVOff+headDim])
		}
	}

	// Reshape Q for attention: [nHead, 1, headDim]
	q = q.Reshape(d.nHead, 1, headDim)

	// Apply scaled dot-product attention (causal).
	attnOut, _, err := ml.ScaledDotProductAttention(ctx, q, kFull, vFull, true, false)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: attention: %w", err)
	}

	// Reshape back: [nHead, 1, headDim] → [1, nTextState]
	// attnOut is [nHead, 1, headDim], concatenate across heads to get [1, nTextState]
	attnOut = attnOut.Reshape(1, d.nHead*headDim)

	// Project output: attnOut @ outW^T + outB → [1, nTextState]
	out, err := ml.MatMulTransB(ctx, attnOut, attn.outW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: out matmul: %w", err)
	}
	out = ml.Add(out, attn.outB)

	return out, nil
}

// runDecoderCrossAttn runs cross-attention to encoder output.
func (d *WhisperDecoder) runDecoderCrossAttn(ctx context.Context, x, encoderOut ml.Tensor, attn decoderCrossAttn, headDim int) (ml.Tensor, error) {
	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: x shape mismatch")
	}

	// Compute Q: x @ qW^T + qB → [1, nTextState]
	q, err := ml.MatMulTransB(ctx, x, attn.qW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: q matmul: %w", err)
	}
	q = ml.Add(q, attn.qB)

	// Compute K from encoder output: encoderOut @ kW^T → [encLen, nTextState]
	k, err := ml.MatMulTransB(ctx, encoderOut, attn.kW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: k matmul: %w", err)
	}

	// Compute V from encoder output: encoderOut @ vW^T + vB → [encLen, nTextState]
	v, err := ml.MatMulTransB(ctx, encoderOut, attn.vW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: v matmul: %w", err)
	}
	v = ml.Add(v, attn.vB)

	// Reshape for attention: Q [nHead, 1, headDim], K/V [nHead, encLen, headDim]
	encLen := len(k.Data) / d.nTextState
	if encLen <= 0 || encLen*d.nTextState != len(k.Data) {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: k shape inconsistent with nTextState")
	}
	q = q.Reshape(d.nHead, 1, headDim)
	k = k.Reshape(d.nHead, encLen, headDim)
	v = v.Reshape(d.nHead, encLen, headDim)

	// Apply scaled dot-product attention (non-causal for encoder output).
	attnOut, _, err := ml.ScaledDotProductAttention(ctx, q, k, v, false, false)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: attention: %w", err)
	}

	// Reshape back: [nHead, 1, headDim] → [1, nTextState]
	// attnOut is [nHead, 1, headDim], concatenate across heads to get [1, nTextState]
	attnOut = attnOut.Reshape(1, d.nHead*headDim)

	// Project output: attnOut @ outW^T + outB → [1, nTextState]
	out, err := ml.MatMulTransB(ctx, attnOut, attn.outW)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: out matmul: %w", err)
	}
	out = ml.Add(out, attn.outB)

	return out, nil
}

// runMLP runs the MLP sub-layer.
func (d *WhisperDecoder) runMLP(ctx context.Context, x ml.Tensor, b *decoderBlock) (ml.Tensor, error) {
	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp: x shape mismatch")
	}

	// MLP.0: x @ mlp0W^T + mlp0B → [1, 4*nTextState]
	out, err := ml.MatMulTransB(ctx, x, b.mlp0W)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp.0: %w", err)
	}
	out = ml.Add(out, b.mlp0B)
	out = ml.GELU(out)

	// MLP.2: out @ mlp2W^T + mlp2B → [1, nTextState]
	out, err = ml.MatMulTransB(ctx, out, b.mlp2W)
	if err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp.2: %w", err)
	}
	out = ml.Add(out, b.mlp2B)

	return out, nil
}

// sampleGreedy samples the next token using greedy sampling with temperature and fallback.
func (d *WhisperDecoder) sampleGreedy(logits []float32, temperature float32, params DecoderParams) (int32, error) {
	if len(logits) != d.nVocab {
		return 0, fmt.Errorf("model: decode: sample: logits length mismatch")
	}

	// Apply temperature scaling.
	scaledLogits := make([]float32, len(logits))
	copy(scaledLogits, logits)
	if temperature > 0 {
		for i := range scaledLogits {
			scaledLogits[i] /= temperature
		}
	}

	// Compute softmax probabilities.
	probs := d.softmax(scaledLogits)

	// Find argmax token and check logprob threshold.
	var maxIdx int
	var maxProb float32
	for i := 0; i < len(probs); i++ {
		if probs[i] > maxProb {
			maxProb = probs[i]
			maxIdx = i
		}
	}

	// Check if top-1 probability is above threshold.
	logprobThold := params.LogprobThold
	if logprobThold > 0 && maxProb > 0 {
		logprob := float32(math.Log(float64(maxProb)))
		if logprob < -logprobThold {
			// Fallback to sampling from top probabilities if enabled.
			if !params.NoFallback && temperature > 0 {
				return d.sampleTopP(probs, 0.9), nil // Default top-p sampling.
			}
		}
	}

	// Check if no-speech token should be suppressed.
	if params.SuppressNST && d.eotToken > 0 {
		if logits[d.eotToken] < -params.NoSpeechThold {
			probs[d.eotToken] = 0
		}
	}

	// Re-find argmax if no-speech was suppressed.
	if params.SuppressNST {
		maxProb = 0
		for i := 0; i < len(probs); i++ {
			if probs[i] > maxProb {
				maxProb = probs[i]
				maxIdx = i
			}
		}
	}

	return int32(maxIdx), nil
}

// topKTokens returns the top-K tokens by probability.
func (d *WhisperDecoder) topKTokens(logits []float32, k int, temperature float32) []struct {
	token   int32
	logprob float32
} {
	if len(logits) != d.nVocab {
		return nil
	}

	// Apply temperature scaling.
	scaledLogits := make([]float32, len(logits))
	copy(scaledLogits, logits)
	if temperature > 0 {
		for i := range scaledLogits {
			scaledLogits[i] /= temperature
		}
	}

	// Compute log-softmax to get log-probabilities.
	maxLogit := scaledLogits[0]
	for i := 1; i < len(scaledLogits); i++ {
		if scaledLogits[i] > maxLogit {
			maxLogit = scaledLogits[i]
		}
	}

	expSum := float32(0)
	for i := range scaledLogits {
		expSum += float32(math.Exp(float64(scaledLogits[i] - maxLogit)))
	}
	logSum := float32(math.Log(float64(expSum))) + maxLogit

	logProbs := make([]float32, len(scaledLogits))
	for i := range scaledLogits {
		logProbs[i] = scaledLogits[i] - logSum
	}

	// Find top-K indices.
	type tokenProb struct {
		token   int32
		logprob float32
	}
	topK := make([]tokenProb, 0, k)
	for i := 0; i < len(logProbs) && len(topK) < k; i++ {
		topK = append(topK, tokenProb{token: int32(i), logprob: logProbs[i]})
	}

	// Simple selection (not fully sorted for efficiency).
	for i := k; i < len(logProbs); i++ {
		minIdx := 0
		for j := 1; j < len(topK); j++ {
			if logProbs[i] > logProbs[int(topK[j].token)] {
				if logProbs[i] < logProbs[int(topK[minIdx].token)] {
					minIdx = j
				}
			}
		}
		if i < len(logProbs) && logProbs[i] > logProbs[int(topK[minIdx].token)] {
			topK[minIdx] = tokenProb{token: int32(i), logprob: logProbs[i]}
		}
	}

	result := make([]struct {
		token   int32
		logprob float32
	}, len(topK))
	for i, tp := range topK {
		result[i].token = tp.token
		result[i].logprob = tp.logprob
	}
	return result
}

// sampleTopP performs top-p (nucleus) sampling.
func (d *WhisperDecoder) sampleTopP(probs []float32, topP float32) int32 {
	if len(probs) == 0 {
		return 0
	}

	// Sort probabilities in descending order.
	type prob struct {
		idx  int
		prob float32
	}
	sorted := make([]prob, len(probs))
	for i := range probs {
		sorted[i] = prob{idx: i, prob: probs[i]}
	}

	// Simple sorting (bubble sort for small sizes).
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].prob > sorted[i].prob {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Accumulate probabilities until reaching top-p.
	var cumSum float32
	for i := 0; i < len(sorted); i++ {
		cumSum += sorted[i].prob
		if cumSum >= topP {
			// Sample uniformly from top-p indices.
			return int32(sorted[rand.Intn(i+1)].idx)
		}
	}

	// Fallback to highest probability.
	return int32(sorted[0].idx)
}

// softmax computes softmax over logits.
func (d *WhisperDecoder) softmax(logits []float32) []float32 {
	maxLogit := logits[0]
	for i := 1; i < len(logits); i++ {
		if logits[i] > maxLogit {
			maxLogit = logits[i]
		}
	}

	probs := make([]float32, len(logits))
	expSum := float32(0)
	for i := range logits {
		probs[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
		expSum += probs[i]
	}

	for i := range probs {
		probs[i] /= expSum
	}

	return probs
}

// tokensToSegments converts a token sequence to segments.
func (d *WhisperDecoder) tokensToSegments(tokens []int32, prompt []int32) ([]Segment, error) {
	// Remove prompt tokens from the result.
	if len(tokens) < len(prompt) {
		return []Segment{}, nil
	}

	resultTokens := tokens[len(prompt):]
	if len(resultTokens) == 0 {
		return []Segment{}, nil
	}

	// Create a single segment with all non-prompt tokens.
	// In a full implementation, this would decode tokens to text and split by timestamps.
	segment := Segment{
		Tokens: make([]TokenData, len(resultTokens)),
	}
	for i, token := range resultTokens {
		segment.Tokens[i] = TokenData{ID: token}
	}

	return []Segment{segment}, nil
}

// sortAndPruneBeams sorts beam candidates by normalized score and returns top-K.
func sortAndPruneBeams(candidates []*beamCandidate, k int) []*beamCandidate {
	if len(candidates) == 0 {
		return []*beamCandidate{}
	}

	// Simple sorting (bubble sort for small sizes).
	for i := 0; i < len(candidates); i++ {
		for j := i + 1; j < len(candidates); j++ {
			if candidates[j].normScore > candidates[i].normScore {
				candidates[i], candidates[j] = candidates[j], candidates[i]
			}
		}
	}

	if k > len(candidates) {
		k = len(candidates)
	}
	return candidates[:k]
}
