package model

import (
	"context"
	"testing"

	"github.com/whispergo/whisper.go/internal/ml"
)

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
