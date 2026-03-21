package model

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/whispergo/whisper.go/internal/ml"
)

var benchmarkSampleTokenSink int32
var benchmarkTopKCountSink int

func BenchmarkDecoderForwardTokenLatency_SoftmaxModes(b *testing.B) {
	ctx := context.Background()
	decoder := buildTestDecoder()

	encLen := 64
	encoderOut := ml.New(encLen, testNTextState)
	for i := range encoderOut.Data {
		encoderOut.Data[i] = 0.01
	}

	makeState := func() *decoderState {
		return decoder.newDecoderState([]int32{50258, 50259, 50357, 50363})
	}

	run := func(b *testing.B) {
		state := makeState()
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			logits, err := decoder.forward(ctx, state, encoderOut)
			if err != nil {
				b.Fatalf("forward failed: %v", err)
			}
			next := int32(i % decoder.nVocab)
			if len(logits) > 0 && logits[0] > 0 {
				next = 0
			}
			state.tokens = append(state.tokens, next)
			if len(state.tokens) >= decoder.nTextCtx {
				state = makeState()
			}
		}
	}

	b.Run("exact_exp", func(b *testing.B) {
		prev := ml.SetFastSoftmaxEnabled(false)
		defer ml.SetFastSoftmaxEnabled(prev)
		run(b)
	})

	b.Run("fast_exp", func(b *testing.B) {
		prev := ml.SetFastSoftmaxEnabled(true)
		defer ml.SetFastSoftmaxEnabled(prev)
		run(b)
	})
}

func BenchmarkDecoderSamplingOnly(b *testing.B) {
	decoder := buildTestDecoder()

	logits := make([]float32, decoder.nVocab)
	for i := range logits {
		logits[i] = -8.0 + 0.00003*float32(i%1000)
	}
	if decoder.nVocab > 20 {
		logits[17] = 7.5
		logits[3] = 6.25
		logits[9] = 5.75
	}

	greedyNoFallback := DecoderParams{
		Temperature:  1.0,
		NoFallback:   true,
		SuppressNST:  false,
		LogprobThold: 0,
	}

	greedyFallback := DecoderParams{
		Temperature:  1.0,
		NoFallback:   false,
		SuppressNST:  false,
		LogprobThold: 1000,
	}

	b.Run("greedy_argmax_path", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tok, err := decoder.sampleGreedy(logits, 1.0, greedyNoFallback)
			if err != nil {
				b.Fatalf("sampleGreedy failed: %v", err)
			}
			benchmarkSampleTokenSink = tok
		}
	})

	b.Run("greedy_fallback_top_p", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tok, err := decoder.sampleGreedy(logits, 1.0, greedyFallback)
			if err != nil {
				b.Fatalf("sampleGreedy fallback failed: %v", err)
			}
			benchmarkSampleTokenSink = tok
		}
	})

	b.Run("top_k_tokens", func(b *testing.B) {
		const k = 5
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			candidates := decoder.topKTokens(logits, k, 1.0)
			if len(candidates) != k {
				b.Fatalf("topKTokens size mismatch: got %d want %d", len(candidates), k)
			}
			benchmarkTopKCountSink += len(candidates)
		}
	})

	b.Run("top_k_tokens_reuse_scratch", func(b *testing.B) {
		const k = 5
		topScratch := make([]tokenLogProb, 0, k)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var candidates []tokenLogProb
			candidates, topScratch = decoder.topKTokensWithScratch(logits, k, 1.0, topScratch)
			if len(candidates) != k {
				b.Fatalf("topKTokensWithScratch size mismatch: got %d want %d", len(candidates), k)
			}
			benchmarkTopKCountSink += len(candidates)
		}
	})

	b.Run("top_p_from_logits", func(b *testing.B) {
		const topP = 0.9
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tok := decoder.sampleTopPFromLogits(logits, 1.0, topP)
			benchmarkSampleTokenSink = tok
		}
	})

	b.Run("top_p_from_logits_reuse_scratch", func(b *testing.B) {
		const topP = 0.9
		idxScratch := make([]int, 0, len(logits))
		scaledScratch := make([]float32, 0, len(logits))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			var tok int32
			tok, idxScratch, scaledScratch = decoder.sampleTopPFromLogitsWithScratch(logits, 1.0, topP, idxScratch, scaledScratch)
			benchmarkSampleTokenSink = tok
		}
	})

	b.Run("top_p_from_probs", func(b *testing.B) {
		const topP = 0.9
		maxLogit := logits[0]
		for i := 1; i < len(logits); i++ {
			if logits[i] > maxLogit {
				maxLogit = logits[i]
			}
		}
		probs := make([]float32, len(logits))
		sum := float32(0)
		for i := range logits {
			probs[i] = float32(math.Exp(float64(logits[i] - maxLogit)))
			sum += probs[i]
		}
		inv := 1 / sum
		for i := range probs {
			probs[i] *= inv
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tok := decoder.sampleTopP(probs, topP)
			benchmarkSampleTokenSink = tok
		}
	})
}

func BenchmarkDecoderSamplingByVocab(b *testing.B) {
	vocabSizes := []int{4096, 16384, 51865}

	for _, vocabSize := range vocabSizes {
		vocabSize := vocabSize
		b.Run(fmt.Sprintf("vocab_%d", vocabSize), func(b *testing.B) {
			decoder := &WhisperDecoder{
				nVocab:   vocabSize,
				eotToken: int32(vocabSize - 1),
			}

			logits := make([]float32, vocabSize)
			for i := range logits {
				logits[i] = -8.0 + 0.00003*float32(i%1000)
			}
			if vocabSize > 20 {
				logits[17] = 7.5
				logits[3] = 6.25
				logits[9] = 5.75
			}

			greedyNoFallback := DecoderParams{
				Temperature:  1.0,
				NoFallback:   true,
				SuppressNST:  false,
				LogprobThold: 0,
			}

			b.Run("greedy_argmax", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					tok, err := decoder.sampleGreedy(logits, 1.0, greedyNoFallback)
					if err != nil {
						b.Fatalf("sampleGreedy failed: %v", err)
					}
					benchmarkSampleTokenSink = tok
				}
			})

			b.Run("top_k_5", func(b *testing.B) {
				const k = 5
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					cands := decoder.topKTokens(logits, k, 1.0)
					if len(cands) != k {
						b.Fatalf("topKTokens size mismatch: got %d want %d", len(cands), k)
					}
					benchmarkTopKCountSink += len(cands)
				}
			})

			b.Run("top_k_5_reuse_scratch", func(b *testing.B) {
				const k = 5
				topScratch := make([]tokenLogProb, 0, k)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var cands []tokenLogProb
					cands, topScratch = decoder.topKTokensWithScratch(logits, k, 1.0, topScratch)
					if len(cands) != k {
						b.Fatalf("topKTokensWithScratch size mismatch: got %d want %d", len(cands), k)
					}
					benchmarkTopKCountSink += len(cands)
				}
			})

			b.Run("top_p_logits_reuse_scratch", func(b *testing.B) {
				const topP = 0.9
				idxScratch := make([]int, 0, len(logits))
				scaledScratch := make([]float32, 0, len(logits))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					var tok int32
					tok, idxScratch, scaledScratch = decoder.sampleTopPFromLogitsWithScratch(logits, 1.0, topP, idxScratch, scaledScratch)
					benchmarkSampleTokenSink = tok
				}
			})
		})
	}
}

func BenchmarkDecoderBeamSearchLatency(b *testing.B) {
	ctx := context.Background()
	decoder := buildTestDecoder()

	encLen := 64
	encoderOut := ml.New(encLen, testNTextState)
	for i := range encoderOut.Data {
		encoderOut.Data[i] = 0.01
	}

	for _, beamSize := range []int{2, 4} {
		beamSize := beamSize
		b.Run(fmt.Sprintf("beam_%d", beamSize), func(b *testing.B) {
			params := DecoderParams{
				Prompt:      []int32{50258, 50259, 50357, 50363},
				BeamSize:    beamSize,
				MaxTokens:   16,
				Temperature: 1.0,
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				segments, err := decoder.decodeBeamSearch(ctx, encoderOut, params)
				if err != nil {
					b.Fatalf("decodeBeamSearch failed: %v", err)
				}
				benchmarkTopKCountSink += len(segments)
			}
		})
	}
}
