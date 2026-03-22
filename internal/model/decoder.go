package model

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/ml"
	"github.com/whispergo/whisper.go/internal/vocab"
)

// decoderCrossAttn holds Q/K/V/out weights for cross-attention to encoder output.
type decoderCrossAttn struct {
	qW, qB     ml.Tensor // [nTextState, nTextState], [nTextState]
	qWQ        *ml.QuantizedMatrix
	kW, vW     ml.Tensor // [nTextState, nTextState] (no bias)
	kWQ, vWQ   *ml.QuantizedMatrix
	vB         ml.Tensor // [nTextState]
	outW, outB ml.Tensor // [nTextState, nTextState], [nTextState]
	outWQ      *ml.QuantizedMatrix
}

// decoderSelfAttn holds Q/K/V/out weights for causal self-attention (with KV cache).
type decoderSelfAttn struct {
	qW, qB     ml.Tensor // [nTextState, nTextState], [nTextState]
	qWQ        *ml.QuantizedMatrix
	kW         ml.Tensor // [nTextState, nTextState] (no bias)
	kWQ        *ml.QuantizedMatrix
	vW, vB     ml.Tensor // [nTextState, nTextState], [nTextState]
	vWQ        *ml.QuantizedMatrix
	outW, outB ml.Tensor // [nTextState, nTextState], [nTextState]
	outWQ      *ml.QuantizedMatrix
}

// decoderBlock holds all weight tensors for one transformer decoder block.
type decoderBlock struct {
	// Self-attention
	sAttnLnW, sAttnLnB ml.Tensor // [nTextState]
	selfAttn           decoderSelfAttn

	// Cross-attention
	cAttnLnW, cAttnLnB ml.Tensor // [nTextState]
	crossAttn          decoderCrossAttn

	// MLP
	mlpLnW, mlpLnB ml.Tensor // [nTextState]
	mlp0W, mlp0B   ml.Tensor // [4*nTextState, nTextState], [4*nTextState]
	mlp0WQ         *ml.QuantizedMatrix
	mlp2W, mlp2B   ml.Tensor // [nTextState, 4*nTextState], [nTextState]
	mlp2WQ         *ml.QuantizedMatrix
}

// WhisperDecoder implements the Decoder interface.
type WhisperDecoder struct {
	backend    ComputeBackend
	nVocab     int
	nTextState int
	nHead      int
	nTextLayer int
	nTextCtx   int

	// Embeddings
	tokenEmb  ml.Tensor // [nVocab, nTextState]
	tokenEmbQ *ml.QuantizedMatrix
	posEmb    ml.Tensor // [nTextCtx, nTextState]

	// Transformer blocks
	blocks []decoderBlock

	// Final layer norm
	lnW, lnB ml.Tensor // [nTextState]

	// Special token IDs
	eotToken int32
	vocab    *vocab.Vocabulary
}

// SetVocabulary attaches vocabulary used to decode token IDs into text.
func (d *WhisperDecoder) SetVocabulary(v *vocab.Vocabulary) {
	d.vocab = v
	if v == nil {
		return
	}
	special := v.Special()
	if special.EOT >= 0 {
		d.eotToken = int32(special.EOT)
	}
}

// NewDecoder loads a decoder from an open GGUF file.
func NewDecoder(f gguf.FileLike) (*WhisperDecoder, error) {
	return NewDecoderWithBackend(f, nil)
}

// NewDecoderWithBackend loads a decoder from an open GGUF file using the provided compute backend.
// If backend is nil, the default ml backend is used.
func NewDecoderWithBackend(f gguf.FileLike, backend ComputeBackend) (*WhisperDecoder, error) {
	ctx := context.Background()
	var err error
	if backend == nil {
		backend = defaultComputeBackend
	}

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
		backend:    backend,
		nVocab:     int(nVocab),
		nTextState: int(nTextState),
		nHead:      int(nHead),
		nTextLayer: int(nLayers),
		nTextCtx:   int(nTextCtx),
		blocks:     make([]decoderBlock, nLayers),
		eotToken:   50256, // Default Whisper EOT token ID
	}

	// Load token embedding [nVocab, nTextState].
	if q, _, qErr := loadQuantizedMatrixAuto(ctx, f, "decoder.token_embedding.weight"); qErr == nil && q != nil && q.Rows == d.nVocab && q.Cols == d.nTextState {
		d.tokenEmbQ = q
	} else {
		if d.tokenEmb, _, err = loadTensorAuto(ctx, f, "decoder.token_embedding.weight"); err != nil {
			return nil, err
		}
		if len(d.tokenEmb.Shape) != 2 {
			return nil, fmt.Errorf("model: decoder: invalid token embedding shape %v", d.tokenEmb.Shape)
		}
		if d.tokenEmb.Shape[0] == d.nTextState && d.tokenEmb.Shape[1] == d.nVocab {
			d.tokenEmb = ml.Transpose(d.tokenEmb, 1, 0)
		}
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
		if b.selfAttn.qW, b.selfAttn.qWQ, _, err = loadMatWeightAuto(ctx, f, p+".self_attn.query.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.selfAttn.qB, _, err = loadTensorAuto(ctx, f, p+".self_attn.query.bias"); err != nil {
			return nil, err
		}
		if b.selfAttn.kW, b.selfAttn.kWQ, _, err = loadMatWeightAuto(ctx, f, p+".self_attn.key.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.selfAttn.vW, b.selfAttn.vWQ, _, err = loadMatWeightAuto(ctx, f, p+".self_attn.value.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.selfAttn.vB, _, err = loadTensorAuto(ctx, f, p+".self_attn.value.bias"); err != nil {
			return nil, err
		}
		if b.selfAttn.outW, b.selfAttn.outWQ, _, err = loadMatWeightAuto(ctx, f, p+".self_attn.out.weight", int(nTextState), int(nTextState)); err != nil {
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
		if b.crossAttn.qW, b.crossAttn.qWQ, _, err = loadMatWeightAuto(ctx, f, p+".cross_attn.query.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.crossAttn.qB, _, err = loadTensorAuto(ctx, f, p+".cross_attn.query.bias"); err != nil {
			return nil, err
		}
		if b.crossAttn.kW, b.crossAttn.kWQ, _, err = loadMatWeightAuto(ctx, f, p+".cross_attn.key.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.crossAttn.vW, b.crossAttn.vWQ, _, err = loadMatWeightAuto(ctx, f, p+".cross_attn.value.weight", int(nTextState), int(nTextState)); err != nil {
			return nil, err
		}
		if b.crossAttn.vB, _, err = loadTensorAuto(ctx, f, p+".cross_attn.value.bias"); err != nil {
			return nil, err
		}
		if b.crossAttn.outW, b.crossAttn.outWQ, _, err = loadMatWeightAuto(ctx, f, p+".cross_attn.out.weight", int(nTextState), int(nTextState)); err != nil {
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

		b.mlp0W, b.mlp0WQ, _, err = loadMatWeightAuto(ctx, f, p+".mlp.0.weight", 4*int(nTextState), int(nTextState))
		if err != nil {
			return nil, err
		}
		if b.mlp0B, _, err = loadTensorAuto(ctx, f, p+".mlp.0.bias"); err != nil {
			return nil, err
		}

		b.mlp2W, b.mlp2WQ, _, err = loadMatWeightAuto(ctx, f, p+".mlp.2.weight", int(nTextState), 4*int(nTextState))
		if err != nil {
			return nil, err
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
	tokens     []int32     // Current token sequence
	logits     []float32   // Latest logits [nVocab]
	kvCache    ml.Tensor   // [nLayer, 2, nHead, nTextCtx, headDim]
	cachedPos  int         // Number of positions cached
	crossK     []ml.Tensor // per-layer cached projected encoder K [encLen, nTextState]
	crossV     []ml.Tensor // per-layer cached projected encoder V [encLen, nTextState]
	crossReady []bool      // whether crossK/crossV are initialized for layer
	scratch    decoderScratch
}

type tokenLogProb struct {
	token   int32
	logprob float32
}

type decoderScratch struct {
	x      ml.Tensor // [1, nTextState]
	xln    ml.Tensor // [1, nTextState]
	q      ml.Tensor // [1, nTextState]
	k      ml.Tensor // [1, nTextState]
	v      ml.Tensor // [1, nTextState]
	attn   ml.Tensor // [nHead, 1, headDim]
	out    ml.Tensor // [1, nTextState]
	mlpH   ml.Tensor // [1, 4*nTextState]
	kFull  ml.Tensor // [nHead, nTextCtx, headDim]
	vFull  ml.Tensor // [nHead, nTextCtx, headDim]
	selfS  ml.Tensor // [nHead, 1, nTextCtx]
	crossS ml.Tensor // [nHead, 1, encLen]
	logits ml.Tensor // [1, nVocab]

	topK   []tokenLogProb
	idxTmp []int
	sTmp   []float32
}

func (d *WhisperDecoder) newDecoderState(prompt []int32) *decoderState {
	headDim := d.nTextState / d.nHead
	state := &decoderState{
		tokens:     make([]int32, len(prompt)),
		kvCache:    ml.New(d.nTextLayer, 2, d.nHead, d.nTextCtx, headDim),
		cachedPos:  0,
		crossK:     make([]ml.Tensor, d.nTextLayer),
		crossV:     make([]ml.Tensor, d.nTextLayer),
		crossReady: make([]bool, d.nTextLayer),
	}
	copy(state.tokens, prompt)

	state.scratch = decoderScratch{
		x:      ml.New(1, d.nTextState),
		xln:    ml.New(1, d.nTextState),
		q:      ml.New(1, d.nTextState),
		k:      ml.New(1, d.nTextState),
		v:      ml.New(1, d.nTextState),
		attn:   ml.New(d.nHead, 1, headDim),
		out:    ml.New(1, d.nTextState),
		mlpH:   ml.New(1, 4*d.nTextState),
		kFull:  ml.New(d.nHead, d.nTextCtx, headDim),
		vFull:  ml.New(d.nHead, d.nTextCtx, headDim),
		selfS:  ml.New(d.nHead, 1, d.nTextCtx),
		logits: ml.New(1, d.nVocab),
	}

	return state
}

func (d *WhisperDecoder) resetDecoderState(state *decoderState, tokens []int32) {
	if cap(state.tokens) < len(tokens) {
		state.tokens = make([]int32, len(tokens))
	} else {
		state.tokens = state.tokens[:len(tokens)]
	}
	copy(state.tokens, tokens)
	for i := range state.kvCache.Data {
		state.kvCache.Data[i] = 0
	}
	state.cachedPos = 0
	for i := range state.crossReady {
		state.crossReady[i] = false
	}
}

func addBiasRowInPlace(dst, bias ml.Tensor) {
	for i := range bias.Data {
		dst.Data[i] += bias.Data[i]
	}
}

func addBiasAndResidualInPlace(dst, add, bias ml.Tensor) {
	for i := range dst.Data {
		dst.Data[i] += add.Data[i] + bias.Data[i]
	}
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
	// Allow a few attempts with increasing temperature if decoder appears stuck
	attempts := 3
	baseTemp := params.Temperature
	if baseTemp < 0 {
		baseTemp = 0
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

	var finalTokens []int32
	for attempt := 0; attempt < attempts; attempt++ {
		state := d.newDecoderState(params.Prompt)
		temperature := baseTemp + float32(attempt)*0.2

		// Simple repetition guard: stop if the same token repeats too many times.
		repeatStreak := 0
		var prevToken int32 = -1

		for len(state.tokens) < len(params.Prompt)+maxTokens {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("model: decode: cancelled: %w", err)
			}

			// Compute logits for the next token.
			logits, err := d.forward(ctx, state, encoderOut)
			if err != nil {
				return nil, fmt.Errorf("model: decode: forward: %w", err)
			}

			// Apply repetition penalty based on recent history before sampling.
			// Make a copy of logits to avoid mutating shared buffers.
			curLogits := make([]float32, len(logits))
			copy(curLogits, logits[:d.nVocab])
			// Tuned values for this model: penalty=2.0, historyWindow=128
			curLogits = applyRepetitionPenalty(curLogits, state.tokens, 2.0, 128)

			// Sample next token using greedy with temperature and fallback.
			nextToken, idxTmp, sTmp, err := d.sampleGreedyWithScratch(curLogits, temperature, params, state.scratch.idxTmp, state.scratch.sTmp)
			if err != nil {
				return nil, fmt.Errorf("model: decode: sample: %w", err)
			}
			state.scratch.idxTmp = idxTmp
			state.scratch.sTmp = sTmp

			state.tokens = append(state.tokens, nextToken)

			// Stop if we hit EOT.
			if nextToken == d.eotToken {
				break
			}

			// Update repetition streak and stop if it grows too large.
			if prevToken == nextToken {
				repeatStreak++
			} else {
				repeatStreak = 1
				prevToken = nextToken
			}
			// Heuristic threshold: 20 identical tokens -> consider decoding stuck.
			if repeatStreak > 20 {
				break
			}
		}

		finalTokens = append([]int32(nil), state.tokens...)

		// Check compression ratio for loop detection.
		if d.vocab != nil {
			// Build non-prompt token list and decode to text.
			promptLen := len(params.Prompt)
			tokIDs := finalTokens
			if len(tokIDs) > promptLen {
				tokIDs = tokIDs[promptLen:]
			} else {
				tokIDs = nil
			}
			var decoded string
			if len(tokIDs) > 0 {
				vtoks := make([]vocab.Token, len(tokIDs))
				for i, t := range tokIDs {
					vtoks[i] = vocab.Token(t)
				}
				decoded = d.vocab.Decode(vtoks)
			}

			if decoded != "" {
				ratio := compressionRatio(decoded)
				if ratio > 2.4 && attempt < attempts-1 && !params.NoFallback {
					// retry with higher temperature
					continue
				}
			}
		}

		break
	}

	if finalTokens == nil {
		finalTokens = []int32{}
	}

	return d.tokensToSegments(finalTokens, params.Prompt)
}

// decodeBeamSearch uses beam search to generate tokens.
func (d *WhisperDecoder) decodeBeamSearch(ctx context.Context, encoderOut ml.Tensor, params DecoderParams) ([]Segment, error) {
	// Beam search with compression-ratio retry and temperature schedule.
	baseTemp := params.Temperature
	if baseTemp < 0 {
		baseTemp = 0
	}
	attempts := 3
	var finalTokens []int32
	for attempt := 0; attempt < attempts; attempt++ {
		temperature := baseTemp + float32(attempt)*0.2

		beamSize := params.BeamSize
		if beamSize <= 0 {
			beamSize = 1
		}

		// Initialize beam hypotheses with reusable decoder state.
		beams := make([]*beam, beamSize)
		for i := 0; i < beamSize; i++ {
			beams[i] = &beam{
				tokens: make([]int32, len(params.Prompt)),
				score:  0,
				state:  d.newDecoderState(params.Prompt),
			}
			copy(beams[i].tokens, params.Prompt)
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

		allCandidates := make([]beamCandidate, 0, beamSize*6)
		reusableBeams := make([]*beam, 0, beamSize)

		for step := 0; step < maxTokens; step++ {
			if err := ctx.Err(); err != nil {
				return nil, fmt.Errorf("model: decode: beam search cancelled: %w", err)
			}

			// Compute logits for all beam hypotheses.
			allCandidates = allCandidates[:0]
			reusableBeams = reusableBeams[:0]

			for _, hyp := range beams {
				reusableBeams = append(reusableBeams, hyp)
				if len(hyp.tokens) > 0 && hyp.tokens[len(hyp.tokens)-1] == d.eotToken {
					// This hypothesis has reached EOT, keep it unchanged.
					normScore := hyp.score / float32(math.Pow(float64(len(hyp.tokens)), 0.6))
					allCandidates = append(allCandidates, beamCandidate{
						parent:    hyp,
						score:     hyp.score,
						token:     d.eotToken,
						logprob:   0,
						normScore: normScore,
					})
					continue
				}

				d.resetDecoderState(hyp.state, hyp.tokens)
				logits, err := d.forward(ctx, hyp.state, encoderOut)
				if err != nil {
					return nil, fmt.Errorf("model: decode: beam search forward: %w", err)
				}

				// Apply repetition penalty per hypothesis to logits before selecting candidates.
				curLogits := make([]float32, d.nVocab)
				copy(curLogits, logits[:d.nVocab])
				// Tuned values for this model: penalty=2.0, historyWindow=128
				curLogits = applyRepetitionPenalty(curLogits, hyp.tokens, 2.0, 128)

				// Get top-K candidates from this hypothesis.
				topK := 5 // Heuristic: keep top 5 candidates per hypothesis.
				candidates, topBuf := d.topKTokensWithScratch(curLogits, topK, temperature, hyp.state.scratch.topK)
				hyp.state.scratch.topK = topBuf

				for _, candidate := range candidates {
					newLen := len(hyp.tokens) + 1
					newScore := hyp.score + candidate.logprob
					// Normalize score by sequence length (length penalty).
					normScore := newScore / float32(math.Pow(float64(newLen), 0.6))
					allCandidates = append(allCandidates, beamCandidate{
						parent:    hyp,
						score:     newScore,
						token:     candidate.token,
						logprob:   candidate.logprob,
						normScore: normScore,
					})
				}
			}

			// Sort by normalized score and keep top-K.
			bestCandidates := sortAndPruneBeams(allCandidates, beamSize)

			// Update beams with top candidates.
			nextBeams := make([]*beam, len(bestCandidates))
			for i, cand := range bestCandidates {
				var nb *beam
				if n := len(reusableBeams); n > 0 {
					nb = reusableBeams[n-1]
					reusableBeams = reusableBeams[:n-1]
				} else {
					nb = &beam{state: d.newDecoderState(nil)}
				}

				if len(cand.parent.tokens) > 0 && cand.parent.tokens[len(cand.parent.tokens)-1] == d.eotToken && cand.token == d.eotToken && cand.logprob == 0 {
					if cap(nb.tokens) < len(cand.parent.tokens) {
						nb.tokens = make([]int32, len(cand.parent.tokens))
					} else {
						nb.tokens = nb.tokens[:len(cand.parent.tokens)]
					}
					copy(nb.tokens, cand.parent.tokens)
				} else {
					newLen := len(cand.parent.tokens) + 1
					if cap(nb.tokens) < newLen {
						nb.tokens = make([]int32, newLen)
					} else {
						nb.tokens = nb.tokens[:newLen]
					}
					copy(nb.tokens, cand.parent.tokens)
					nb.tokens[newLen-1] = cand.token
				}

				nb.score = cand.score
				nextBeams[i] = nb
			}
			beams = nextBeams

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

			// Safety: if best beam shows excessive repetition, assume stuck and stop.
			if len(beams) > 0 {
				best := beams[0]
				if len(best.tokens) >= 2 {
					last := best.tokens[len(best.tokens)-1]
					streak := 1
					for i := len(best.tokens) - 2; i >= 0 && best.tokens[i] == last; i-- {
						streak++
						if streak > 20 {
							allEOT = true
							break
						}
					}
					if allEOT {
						break
					}
				}
			}
		}

		// Return segments from the best beam (first hypothesis).
		if len(beams) == 0 {
			finalTokens = []int32{}
		} else {
			finalTokens = append([]int32(nil), beams[0].tokens...)
		}

		// If vocabulary is available, check compression ratio and retry with higher temperature if stuck.
		if d.vocab != nil && len(finalTokens) > len(params.Prompt) {
			// Extract non-prompt tokens and decode text.
			tokIDs := finalTokens[len(params.Prompt):]
			vtoks := make([]vocab.Token, 0, len(tokIDs))
			for _, t := range tokIDs {
				vtoks = append(vtoks, vocab.Token(t))
			}
			decoded := d.vocab.Decode(vtoks)
			if decoded != "" {
				ratio := compressionRatio(decoded)
				if ratio > 2.4 && attempt < attempts-1 && !params.NoFallback {
					// retry with higher temperature
					continue
				}
			}
		}

		break
	}

	if finalTokens == nil {
		finalTokens = []int32{}
	}
	return d.tokensToSegments(finalTokens, params.Prompt)
}

// beam represents a hypothesis in beam search.
type beam struct {
	tokens []int32
	score  float32
	state  *decoderState
}

// beamCandidate represents a candidate token for beam expansion.
type beamCandidate struct {
	parent    *beam
	score     float32
	token     int32
	logprob   float32
	normScore float32
}

// forward runs the decoder forward pass for the next token.
func (d *WhisperDecoder) forward(ctx context.Context, state *decoderState, encoderOut ml.Tensor) ([]float32, error) {
	backend := d.computeBackend()

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

	// x = token_embedding + positional_embedding [1, nTextState]
	x := state.scratch.x
	copy(x.Data, d.tokenEmb.Data[startIdx:endIdx])
	posStartIdx := nextPos * d.nTextState
	for i := 0; i < d.nTextState; i++ {
		x.Data[i] += d.posEmb.Data[posStartIdx+i]
	}

	headDim := d.nTextState / d.nHead

	// Pass through decoder blocks.
	for i := range d.blocks {
		b := &d.blocks[i]

		// Self-attention with KV cache.
		xln := state.scratch.xln
		if err := backend.LayerNormInto(xln, x, b.sAttnLnW, b.sAttnLnB, layerNormEps); err != nil {
			return nil, fmt.Errorf("model: decode: block %d self_attn layernorm: %w", i, err)
		}
		attnOut, err := d.runDecoderSelfAttn(ctx, state, xln, b.selfAttn, nextPos, i, headDim)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d self_attn: %w", i, err)
		}
		addBiasAndResidualInPlace(x, attnOut, b.selfAttn.outB)

		// Cross-attention to encoder output.
		if err := backend.LayerNormInto(xln, x, b.cAttnLnW, b.cAttnLnB, layerNormEps); err != nil {
			return nil, fmt.Errorf("model: decode: block %d cross_attn layernorm: %w", i, err)
		}
		crossOut, err := d.runDecoderCrossAttn(ctx, state, xln, encoderOut, b.crossAttn, i, headDim)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d cross_attn: %w", i, err)
		}
		addBiasAndResidualInPlace(x, crossOut, b.crossAttn.outB)

		// MLP sub-layer.
		if err := backend.LayerNormInto(xln, x, b.mlpLnW, b.mlpLnB, layerNormEps); err != nil {
			return nil, fmt.Errorf("model: decode: block %d mlp layernorm: %w", i, err)
		}
		mlpOut, err := d.runMLP(ctx, state, xln, b)
		if err != nil {
			return nil, fmt.Errorf("model: decode: block %d mlp: %w", i, err)
		}
		addBiasAndResidualInPlace(x, mlpOut, b.mlp2B)

		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("model: decode: cancelled after block %d: %w", i, err)
		}
	}

	// Final layer norm.
	if err := backend.LayerNormInto(x, x, d.lnW, d.lnB, layerNormEps); err != nil {
		return nil, fmt.Errorf("model: decode: forward: final layernorm: %w", err)
	}

	// Project to vocabulary: x @ tokenEmb^T → [1, nVocab]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, d.tokenEmb, d.tokenEmbQ, state.scratch.logits); err != nil {
		return nil, fmt.Errorf("model: decode: forward: logits: %w", err)
	}
	logits := state.scratch.logits

	if len(logits.Shape) != 2 || logits.Shape[0] != 1 || logits.Shape[1] != d.nVocab {
		return nil, fmt.Errorf("model: decode: forward: logits shape mismatch: %v", logits.Shape)
	}

	return logits.Data[:d.nVocab], nil
}

// runDecoderSelfAttn runs causal self-attention with KV cache.
func (d *WhisperDecoder) runDecoderSelfAttn(ctx context.Context, state *decoderState, x ml.Tensor, attn decoderSelfAttn, pos int, layerIdx, headDim int) (ml.Tensor, error) {
	backend := d.computeBackend()

	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: x shape mismatch")
	}
	kvCache := state.kvCache
	sc := &state.scratch

	// Compute Q: x @ qW^T + qB → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, attn.qW, attn.qWQ, sc.q); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: q matmul: %w", err)
	}
	addBiasRowInPlace(sc.q, attn.qB)

	// Compute K: x @ kW^T → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, attn.kW, attn.kWQ, sc.k); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: k matmul: %w", err)
	}

	// Compute V: x @ vW^T + vB → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, attn.vW, attn.vWQ, sc.v); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: v matmul: %w", err)
	}
	addBiasRowInPlace(sc.v, attn.vB)

	// Update KV cache with new K and V.
	// kvCache shape: [nLayer, 2, nHead, nTextCtx, headDim]
	nTextCtx := d.nTextCtx
	headBlock := nTextCtx * headDim
	base := layerIdx * 2 * d.nHead * headBlock
	for h := 0; h < d.nHead; h++ {
		src := h * headDim
		kOff := base + h*headBlock + pos*headDim
		vOff := base + d.nHead*headBlock + h*headBlock + pos*headDim
		copy(kvCache.Data[kOff:kOff+headDim], sc.k.Data[src:src+headDim])
		copy(kvCache.Data[vOff:vOff+headDim], sc.v.Data[src:src+headDim])
	}

	// Retrieve accumulated K and V up to pos+1.
	// We need [nHead, pos+1, headDim] for both.
	seqLen := pos + 1
	kFull := ml.From(sc.kFull.Data[:d.nHead*seqLen*headDim], d.nHead, seqLen, headDim)
	vFull := ml.From(sc.vFull.Data[:d.nHead*seqLen*headDim], d.nHead, seqLen, headDim)

	kBase := layerIdx * 2 * d.nHead * headBlock
	vBase := kBase + d.nHead*headBlock
	for h := 0; h < d.nHead; h++ {
		srcK := kBase + h*headBlock
		dstK := h * seqLen * headDim
		copy(kFull.Data[dstK:dstK+seqLen*headDim], kvCache.Data[srcK:srcK+seqLen*headDim])

		srcV := vBase + h*headBlock
		dstV := h * seqLen * headDim
		copy(vFull.Data[dstV:dstV+seqLen*headDim], kvCache.Data[srcV:srcV+seqLen*headDim])
	}

	// Reshape Q for attention: [nHead, 1, headDim]
	q := ml.From(sc.q.Data, d.nHead, 1, headDim)

	// Apply scaled dot-product attention (causal) into scratch buffers.
	if err := backend.ScaledDotProductAttentionInto(ctx, q, kFull, vFull, true, sc.attn, ml.Tensor{}, ml.From(sc.selfS.Data[:d.nHead*seqLen], d.nHead, 1, seqLen)); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: attention: %w", err)
	}

	// Reshape back: [nHead, 1, headDim] -> [1, nTextState]
	attnOut := sc.attn.Reshape(1, d.nHead*headDim)

	// Project output: attnOut @ outW^T + outB → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, attnOut, attn.outW, attn.outWQ, sc.out); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: self_attn: out matmul: %w", err)
	}

	return sc.out, nil
}

// runDecoderCrossAttn runs cross-attention to encoder output.
func (d *WhisperDecoder) runDecoderCrossAttn(ctx context.Context, state *decoderState, x, encoderOut ml.Tensor, attn decoderCrossAttn, layerIdx, headDim int) (ml.Tensor, error) {
	backend := d.computeBackend()

	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: x shape mismatch")
	}
	sc := &state.scratch

	// Compute Q: x @ qW^T + qB → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, attn.qW, attn.qWQ, sc.q); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: q matmul: %w", err)
	}
	addBiasRowInPlace(sc.q, attn.qB)

	if !state.crossReady[layerIdx] {
		encLen := encoderOut.Shape[0]
		state.crossK[layerIdx] = ml.New(encLen, d.nTextState)
		state.crossV[layerIdx] = ml.New(encLen, d.nTextState)
		if err := matMulTransBMaybeQuantInto(ctx, backend, encoderOut, attn.kW, attn.kWQ, state.crossK[layerIdx]); err != nil {
			return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: k matmul: %w", err)
		}
		if err := matMulTransBMaybeQuantInto(ctx, backend, encoderOut, attn.vW, attn.vWQ, state.crossV[layerIdx]); err != nil {
			return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: v matmul: %w", err)
		}
		for i := 0; i < encLen; i++ {
			base := i * d.nTextState
			for j := 0; j < d.nTextState; j++ {
				state.crossV[layerIdx].Data[base+j] += attn.vB.Data[j]
			}
		}
		state.crossReady[layerIdx] = true
	}
	k := state.crossK[layerIdx]
	v := state.crossV[layerIdx]

	// Reshape for attention: Q [nHead, 1, headDim], K/V [nHead, encLen, headDim]
	encLen := len(k.Data) / d.nTextState
	if encLen <= 0 || encLen*d.nTextState != len(k.Data) {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: k shape inconsistent with nTextState")
	}
	q := ml.From(sc.q.Data, d.nHead, 1, headDim)
	k = k.Reshape(d.nHead, encLen, headDim)
	v = v.Reshape(d.nHead, encLen, headDim)

	if len(sc.crossS.Shape) == 0 || sc.crossS.Shape[0] != d.nHead || sc.crossS.Shape[2] < encLen {
		sc.crossS = ml.New(d.nHead, 1, encLen)
	}

	// Apply scaled dot-product attention (non-causal for encoder output) into scratch buffers.
	if err := backend.ScaledDotProductAttentionInto(ctx, q, k, v, false, sc.attn, ml.Tensor{}, ml.From(sc.crossS.Data[:d.nHead*encLen], d.nHead, 1, encLen)); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: attention: %w", err)
	}

	// Reshape back: [nHead, 1, headDim] -> [1, nTextState]
	attnOut := sc.attn.Reshape(1, d.nHead*headDim)

	// Project output: attnOut @ outW^T + outB → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, attnOut, attn.outW, attn.outWQ, sc.out); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: cross_attn: out matmul: %w", err)
	}

	return sc.out, nil
}

// runMLP runs the MLP sub-layer.
func (d *WhisperDecoder) runMLP(ctx context.Context, state *decoderState, x ml.Tensor, b *decoderBlock) (ml.Tensor, error) {
	backend := d.computeBackend()

	if len(x.Shape) != 2 || x.Shape[1] != d.nTextState {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp: x shape mismatch")
	}
	sc := &state.scratch

	// MLP.0: x @ mlp0W^T + mlp0B → [1, 4*nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, x, b.mlp0W, b.mlp0WQ, sc.mlpH); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp.0: %w", err)
	}
	addBiasRowInPlace(sc.mlpH, b.mlp0B)
	backend.GELUInPlace(sc.mlpH)

	// MLP.2: out @ mlp2W^T + mlp2B → [1, nTextState]
	if err := matMulTransBMaybeQuantInto(ctx, backend, sc.mlpH, b.mlp2W, b.mlp2WQ, sc.out); err != nil {
		return ml.Tensor{}, fmt.Errorf("model: decode: mlp.2: %w", err)
	}

	return sc.out, nil
}

// sampleGreedy samples the next token using greedy sampling with temperature and fallback.
func (d *WhisperDecoder) sampleGreedyWithScratch(logits []float32, temperature float32, params DecoderParams, idxScratch []int, scaledScratch []float32) (int32, []int, []float32, error) {
	if len(logits) != d.nVocab {
		return 0, idxScratch, scaledScratch, fmt.Errorf("model: decode: sample: logits length mismatch")
	}

	invTemp := float32(1.0)
	if temperature > 0 {
		invTemp = 1 / temperature
	}

	maxIdx := 0
	maxScaled := logits[0] * invTemp
	for i := 1; i < len(logits); i++ {
		s := logits[i] * invTemp
		if s > maxScaled {
			maxScaled = s
			maxIdx = i
		}
	}

	if params.SuppressNST && d.eotToken >= 0 && int(d.eotToken) < len(logits) {
		if logits[d.eotToken] < -params.NoSpeechThold && maxIdx == int(d.eotToken) {
			maxIdx = -1
			maxScaled = float32(math.Inf(-1))
			for i := 0; i < len(logits); i++ {
				if i == int(d.eotToken) {
					continue
				}
				s := logits[i] * invTemp
				if s > maxScaled {
					maxScaled = s
					maxIdx = i
				}
			}
			if maxIdx < 0 {
				maxIdx = int(d.eotToken)
			}
		}
	}

	logprobThold := params.LogprobThold
	if logprobThold > 0 && !params.NoFallback && temperature > 0 {
		scaledMax := float32(math.Inf(-1))
		for i := 0; i < len(logits); i++ {
			s := logits[i] * invTemp
			if s > scaledMax {
				scaledMax = s
			}
		}

		expSum := float32(0)
		for i := 0; i < len(logits); i++ {
			expSum += float32(math.Exp(float64(logits[i]*invTemp - scaledMax)))
		}
		logZ := float32(math.Log(float64(expSum))) + scaledMax
		maxLogProb := logits[maxIdx]*invTemp - logZ
		if maxLogProb < -logprobThold {
			tok, idxScratch, scaledScratch := d.sampleTopPFromLogitsWithScratch(logits, temperature, 0.9, idxScratch, scaledScratch)
			return tok, idxScratch, scaledScratch, nil
		}
	}

	return int32(maxIdx), idxScratch, scaledScratch, nil
}

// sampleGreedy samples the next token using greedy sampling with temperature and fallback.
func (d *WhisperDecoder) sampleGreedy(logits []float32, temperature float32, params DecoderParams) (int32, error) {
	tok, _, _, err := d.sampleGreedyWithScratch(logits, temperature, params, nil, nil)
	return tok, err
}

// topKTokens returns the top-K tokens by probability.
func (d *WhisperDecoder) topKTokensWithScratch(logits []float32, k int, temperature float32, top []tokenLogProb) ([]tokenLogProb, []tokenLogProb) {
	if len(logits) != d.nVocab || k <= 0 {
		return nil, top
	}
	vals := logits[:len(logits):len(logits)]
	if k > len(logits) {
		k = len(vals)
	}

	invTemp := float32(1.0)
	if temperature > 0 {
		invTemp = 1 / temperature
	}

	if cap(top) < k {
		top = make([]tokenLogProb, 0, k)
	} else {
		top = top[:0]
	}
	minPos := -1
	minVal := float32(0)

	scaledMax := float32(math.Inf(-1))
	for i, lg := range vals {
		s := lg * invTemp
		if s > scaledMax {
			scaledMax = s
		}

		if len(top) < k {
			top = append(top, tokenLogProb{token: int32(i), logprob: s})
			if minPos < 0 || s < minVal {
				minPos = len(top) - 1
				minVal = s
			}
			continue
		}

		if s <= minVal {
			continue
		}

		top[minPos] = tokenLogProb{token: int32(i), logprob: s}
		minPos = 0
		minVal = top[0].logprob
		for j := 1; j < len(top); j++ {
			if top[j].logprob < minVal {
				minVal = top[j].logprob
				minPos = j
			}
		}
	}

	expSum := float32(0)
	for _, lg := range vals {
		expSum += float32(math.Exp(float64(lg*invTemp - scaledMax)))
	}
	logZ := float32(math.Log(float64(expSum))) + scaledMax

	for i := range top {
		top[i].logprob -= logZ
	}

	sort.Slice(top, func(i, j int) bool {
		return top[i].logprob > top[j].logprob
	})

	return top, top
}

// applyRepetitionPenalty returns a modified copy of logits where tokens appearing
// in recent history have their logits decreased by penalty * count.
func applyRepetitionPenalty(logits []float32, history []int32, penalty float32, historyWindow int) []float32 {
	if penalty <= 0 || len(history) == 0 || historyWindow <= 0 {
		return logits
	}
	n := len(logits)
	out := make([]float32, n)
	copy(out, logits)

	// Consider only last `historyWindow` tokens of history.
	start := 0
	if len(history) > historyWindow {
		start = len(history) - historyWindow
	}
	counts := make(map[int]int)
	for i := start; i < len(history); i++ {
		id := int(history[i])
		if id >= 0 && id < n {
			counts[id]++
		}
	}
	for id, cnt := range counts {
		if cnt <= 0 || id < 0 || id >= n {
			continue
		}
		out[id] -= penalty * float32(cnt)
	}
	return out
}

// compressionRatio returns original_length / compressed_length using gzip.
func compressionRatio(s string) float32 {
	if s == "" {
		return 0
	}
	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	_, _ = io.Copy(gw, strings.NewReader(s))
	_ = gw.Close()
	comp := buf.Len()
	if comp == 0 {
		return float32(len(s))
	}
	return float32(len(s)) / float32(comp)
}

// collapseRepeatedSubstrings detects short repeated units in `s` and collapses
// runs where the same unit repeats >= minRepeats times. This helps remove
// BPE-level repetition like "tra tra tra ..." when merged to text.
func collapseRepeatedSubstrings(s string, maxUnitLen, minRepeats int) string {
	if s == "" || maxUnitLen <= 0 || minRepeats <= 1 {
		return s
	}
	n := len(s)
	// For simplicity operate on bytes (UTF-8 may split, but repeated ASCII runs are common in BPE loops).
	b := []byte(s)
	out := make([]byte, 0, n)
	i := 0
	for i < n {
		// Try unit lengths from 1..maxUnitLen (but bounded by remaining length)
		maxL := maxUnitLen
		if maxL > n-i {
			maxL = n - i
		}
		collapsed := false
		for l := 1; l <= maxL; l++ {
			// Check how many times b[i:i+l] repeats starting at i
			unit := b[i : i+l]
			cnt := 1
			j := i + l
			for j+l <= n && bytes.Equal(b[j:j+l], unit) {
				cnt++
				j += l
				if cnt >= minRepeats {
					break
				}
			}
			if cnt >= minRepeats {
				// append one unit and skip the repeats
				out = append(out, unit...)
				i += l * cnt
				collapsed = true
				break
			}
		}
		if !collapsed {
			out = append(out, b[i])
			i++
		}
	}
	return string(out)
}

// topKTokens returns the top-K tokens by probability.
func (d *WhisperDecoder) topKTokens(logits []float32, k int, temperature float32) []struct {
	token   int32
	logprob float32
} {
	top, _ := d.topKTokensWithScratch(logits, k, temperature, nil)
	result := make([]struct {
		token   int32
		logprob float32
	}, len(top))
	for i := range top {
		result[i].token = top[i].token
		result[i].logprob = top[i].logprob
	}
	return result
}

// sampleTopPFromLogitsWithScratch performs top-p (nucleus) sampling from logits.
// It reuses caller-provided scratch slices when capacity is sufficient.
func (d *WhisperDecoder) sampleTopPFromLogitsWithScratch(logits []float32, temperature, topP float32, idxScratch []int, scaledScratch []float32) (int32, []int, []float32) {
	if len(logits) == 0 {
		return 0, idxScratch, scaledScratch
	}

	invTemp := float32(1.0)
	if temperature > 0 {
		invTemp = 1 / temperature
	}

	n := len(logits)
	if cap(idxScratch) < n {
		idxScratch = make([]int, n)
	} else {
		idxScratch = idxScratch[:n]
	}
	if cap(scaledScratch) < n {
		scaledScratch = make([]float32, n)
	} else {
		scaledScratch = scaledScratch[:n]
	}

	scaledMax := logits[0] * invTemp
	for i, lg := range logits {
		s := lg * invTemp
		idxScratch[i] = i
		scaledScratch[i] = s
		if s > scaledMax {
			scaledMax = s
		}
	}

	expSum := float32(0)
	for _, s := range scaledScratch {
		expSum += float32(math.Exp(float64(s - scaledMax)))
	}
	logZ := float32(math.Log(float64(expSum))) + scaledMax

	const nucleusFastLimit = 512
	limit := nucleusFastLimit
	if limit > n {
		limit = n
	}

	selected := idxScratch[:0]
	minPos := -1
	minScore := float32(0)
	for i := 0; i < n; i++ {
		if len(selected) < limit {
			selected = append(selected, i)
			s := scaledScratch[i]
			if minPos < 0 || s < minScore {
				minPos = len(selected) - 1
				minScore = s
			}
			continue
		}
		s := scaledScratch[i]
		if s > minScore {
			selected[minPos] = i
			minPos = 0
			minScore = scaledScratch[selected[0]]
			for j := 1; j < len(selected); j++ {
				sj := scaledScratch[selected[j]]
				if sj < minScore {
					minScore = sj
					minPos = j
				}
			}
		}
	}

	sort.Slice(selected, func(i, j int) bool {
		return scaledScratch[selected[i]] > scaledScratch[selected[j]]
	})

	topCount := 0
	var cumSum float32
	for i := 0; i < len(selected); i++ {
		s := scaledScratch[selected[i]]
		p := float32(math.Exp(float64(s - logZ)))
		cumSum += p
		topCount = i + 1
		if cumSum >= topP {
			break
		}
	}

	// Fallback to full sort only when the limited candidate set cannot hit top-p.
	if cumSum < topP && len(selected) < n {
		for i := 0; i < n; i++ {
			idxScratch[i] = i
		}
		sort.Slice(idxScratch, func(i, j int) bool {
			return scaledScratch[idxScratch[i]] > scaledScratch[idxScratch[j]]
		})
		selected = idxScratch
		topCount = 0
		cumSum = 0
		for i := 0; i < len(selected); i++ {
			s := scaledScratch[selected[i]]
			p := float32(math.Exp(float64(s - logZ)))
			cumSum += p
			topCount = i + 1
			if cumSum >= topP {
				break
			}
		}
	}

	if topCount == 0 {
		return int32(selected[0]), idxScratch, scaledScratch
	}

	r := rand.Float32() * cumSum
	var run float32
	for i := 0; i < topCount; i++ {
		idx := selected[i]
		s := scaledScratch[idx]
		p := float32(math.Exp(float64(s - logZ)))
		run += p
		if run >= r {
			return int32(idx), idxScratch, scaledScratch
		}
	}

	return int32(selected[topCount-1]), idxScratch, scaledScratch
}

// sampleTopPFromLogits performs top-p (nucleus) sampling from logits.
func (d *WhisperDecoder) sampleTopPFromLogits(logits []float32, temperature, topP float32) int32 {
	tok, _, _ := d.sampleTopPFromLogitsWithScratch(logits, temperature, topP, nil, nil)
	return tok
}

// sampleTopP performs top-p (nucleus) sampling from probabilities.
func (d *WhisperDecoder) sampleTopP(probs []float32, topP float32) int32 {
	if len(probs) == 0 {
		return 0
	}
	type prob struct {
		idx  int
		prob float32
	}
	sorted := make([]prob, len(probs))
	for i := range probs {
		sorted[i] = prob{idx: i, prob: probs[i]}
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].prob > sorted[j].prob
	})
	var cumSum float32
	for i := 0; i < len(sorted); i++ {
		cumSum += sorted[i].prob
		if cumSum >= topP {
			return int32(sorted[rand.Intn(i+1)].idx)
		}
	}

	// Fallback to highest probability.
	return int32(sorted[0].idx)
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
		Tokens: make([]TokenData, 0, len(resultTokens)),
	}
	// Decode tokens individually and collapse consecutive identical subword tokens.
	var lastText string
	decodedTokens := make([]vocab.Token, 0, len(resultTokens))
	for _, token := range resultTokens {
		tok := TokenData{ID: token}
		if d.vocab != nil {
			tok.Text = d.vocab.DecodeToken(vocab.Token(token))
		}
		// Collapse consecutive identical subword tokens (avoid long repeated BPE runs).
		if tok.Text != "" && tok.Text == lastText {
			// skip appending duplicate consecutive token
			continue
		}
		lastText = tok.Text
		segment.Tokens = append(segment.Tokens, tok)
		if d.vocab != nil {
			decodedTokens = append(decodedTokens, vocab.Token(token))
		}
	}

	if d.vocab != nil {
		// Rebuild segment text from filtered token sequence and collapse repeated substrings.
		segment.Text = d.vocab.Decode(decodedTokens)
		// Collapse obvious BPE-level repetition runs (tuned parameters).
		segment.Text = collapseRepeatedSubstrings(segment.Text, 6, 3)
	}

	return []Segment{segment}, nil
}

// sortAndPruneBeams sorts beam candidates by normalized score and returns top-K.
func sortAndPruneBeams(candidates []beamCandidate, k int) []beamCandidate {
	if len(candidates) == 0 {
		return []beamCandidate{}
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].normScore > candidates[j].normScore
	})

	if k > len(candidates) {
		k = len(candidates)
	}
	return candidates[:k]
}
