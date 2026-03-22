package model

import (
	"context"
	"fmt"
	"time"

	"github.com/whispergo/whisper.go/internal/audio"
	"github.com/whispergo/whisper.go/internal/ml"
	"github.com/whispergo/whisper.go/internal/vad"
	"github.com/whispergo/whisper.go/internal/vocab"
)

// WhisperPipeline implements the Transcriber interface.
type WhisperPipeline struct {
	encoder Encoder
	decoder Decoder
	vocab   *vocab.Vocabulary
	vad     vad.VAD
	dtw     DTWAligner
	mel     audio.MelFilters
}

// New creates a pipeline from encoder, decoder, vocabulary, and optional VAD/DTW.
func New(enc Encoder, dec Decoder, v *vocab.Vocabulary, vadModel vad.VAD, dtwAligner DTWAligner) (*WhisperPipeline, error) {
	if enc == nil || dec == nil || v == nil {
		return nil, fmt.Errorf("pipeline: encoder, decoder, and vocabulary are required")
	}
	return &WhisperPipeline{
		encoder: enc,
		decoder: dec,
		vocab:   v,
		vad:     vadModel,
		dtw:     dtwAligner,
		mel:     defaultMelFilters(audio.NMel),
	}, nil
}

// SetMelFilters configures the mel filterbank used by audio preprocessing.
func (p *WhisperPipeline) SetMelFilters(filters audio.MelFilters) {
	if len(filters.Data) == 0 {
		return
	}
	p.mel = filters
}

func defaultMelFilters(nMel int) audio.MelFilters {
	nFreq := audio.NFFT/2 + 1
	filters := audio.MelFilters{NMel: nMel, Data: make([]float32, nMel*nFreq)}
	for m := 0; m < nMel; m++ {
		k := m * nFreq / nMel
		if k < nFreq {
			filters.Data[m*nFreq+k] = 1.0
		}
	}
	return filters
}

// Transcribe runs the full inference loop over chunked audio.
func (p *WhisperPipeline) Transcribe(ctx context.Context, samples []float32, params TranscribeParams) (*Result, error) {
	startAll := time.Now()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	if params.Logger != nil {
		params.Logger.Info("starting transcription", "samples", len(samples))
	}

	if len(samples) == 0 {
		return &Result{Segments: []Segment{}, Language: ""}, nil
	}

	// Constants for chunking
	windowSamples := audio.ChunkSize * audio.SampleRate // 30 seconds * 16000 Hz = 480000 samples
	hopSamples := windowSamples                         // non-overlapping 30-second windows

	// Compute chunks
	numChunks := (len(samples) + hopSamples - 1) / hopSamples
	if numChunks == 0 {
		return &Result{Segments: []Segment{}, Language: ""}, nil
	}

	result := &Result{
		Segments: []Segment{},
		Language: "",
	}
	var firstChunkEncoderOut ml.Tensor
	firstChunkEncoderReady := false

	// Language detection on first chunk
	if params.AutoDetectLanguage {
		langStart := time.Now()
		lang, encOut, err := p.detectLanguage(ctx, samples[:minInt(len(samples), windowSamples)], params)
		if err != nil {
			if params.Logger != nil {
				params.Logger.Error("language detection failed", "error", err)
			}
			return nil, fmt.Errorf("pipeline: language detection: %w", err)
		}
		result.Language = lang
		firstChunkEncoderOut = encOut
		firstChunkEncoderReady = true
		if params.Logger != nil {
			params.Logger.Info("language detected", "lang", lang, "ms", time.Since(langStart).Milliseconds())
		}
	} else if params.Language != "" {
		result.Language = params.Language
	}

	// Process each chunk
	for chunkIdx := 0; chunkIdx < numChunks; chunkIdx++ {
		chunkStart := time.Now()
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}

		if params.Logger != nil {
			params.Logger.Debug("processing chunk", "index", chunkIdx, "total", numChunks)
		}

		// Extract chunk with overlap
		startSample := chunkIdx * hopSamples
		endSample := minInt(startSample+windowSamples, len(samples))
		chunkSamples := samples[startSample:endSample]

		// Pad if necessary
		if len(chunkSamples) < windowSamples {
			padded := make([]float32, windowSamples)
			copy(padded, chunkSamples)
			chunkSamples = padded
		}

		// VAD filtering (optional)
		if params.VADEnabled && p.vad != nil {
			segments, err := p.vad.Detect(ctx, chunkSamples, audio.SampleRate, vad.DefaultParams())
			if err != nil {
				return nil, fmt.Errorf("pipeline: vad detection: %w", err)
			}
			if len(segments) == 0 {
				if params.Logger != nil {
					params.Logger.Debug("no speech detected in chunk", "index", chunkIdx)
				}
				continue
			}
		}

		// Compute mel-spectrogram with model-native filterbank.
		melStart := time.Now()
		mel, err := audio.LogMel(ctx, chunkSamples, p.mel)
		if err != nil {
			return nil, fmt.Errorf("pipeline: mel spectrogram: %w", err)
		}
		melMs := time.Since(melStart).Milliseconds()

		// Convert mel to tensor
		melTensor := ml.New(mel.NMel, mel.NLen)
		melTensor.Data = mel.Data

		// Encode
		encStart := time.Now()
		encReused := false
		var encoderOut ml.Tensor
		if chunkIdx == 0 && firstChunkEncoderReady {
			encoderOut = firstChunkEncoderOut
			encReused = true
		} else {
			encoderOut, err = p.encoder.Encode(ctx, melTensor)
			if err != nil {
				return nil, fmt.Errorf("pipeline: encoder: %w", err)
			}
		}
		encMs := time.Since(encStart).Milliseconds()

		// Prepare decoder prompt
		prompt := p.buildPrompt(result.Language, params, chunkIdx == 0)

		// Decode
		decoderParams := params.DecoderParams
		decoderParams.Prompt = prompt

		decStart := time.Now()
		segments, err := p.decoder.Decode(ctx, encoderOut, decoderParams)
		if err != nil {
			return nil, fmt.Errorf("pipeline: decoder: %w", err)
		}
		decMs := time.Since(decStart).Milliseconds()

		// Post-process segments
		postStart := time.Now()
		processedSegments, err := p.postProcessSegments(ctx, segments, params, int64(startSample)*1000/int64(audio.SampleRate))
		if err != nil {
			return nil, fmt.Errorf("pipeline: post-processing: %w", err)
		}
		postMs := time.Since(postStart).Milliseconds()
		chunkMs := time.Since(chunkStart).Milliseconds()

		if params.Logger != nil && len(processedSegments) > 0 {
			params.Logger.Info("chunk transcribed", "index", chunkIdx, "segments", len(processedSegments), "mel_ms", melMs, "enc_ms", encMs, "enc_reused", encReused, "dec_ms", decMs, "post_ms", postMs, "chunk_ms", chunkMs)
		}

		result.Segments = append(result.Segments, processedSegments...)
	}

	if params.Logger != nil {
		params.Logger.Info("transcription finished", "chunks", numChunks, "segments", len(result.Segments), "total_ms", time.Since(startAll).Milliseconds())
	}

	return result, nil
}

// detectLanguage detects the language from the first chunk.
func (p *WhisperPipeline) detectLanguage(ctx context.Context, samples []float32, params TranscribeParams) (string, ml.Tensor, error) {
	// Ensure samples are exactly windowSamples long
	windowSamples := audio.ChunkSize * audio.SampleRate
	if len(samples) < windowSamples {
		padded := make([]float32, windowSamples)
		copy(padded, samples)
		samples = padded
	} else if len(samples) > windowSamples {
		samples = samples[:windowSamples]
	}

	// Compute mel-spectrogram with model-native filterbank.
	mel, err := audio.LogMel(ctx, samples, p.mel)
	if err != nil {
		return "", ml.Tensor{}, fmt.Errorf("language detection: mel spectrogram: %w", err)
	}

	// Convert mel to tensor
	melTensor := ml.New(mel.NMel, mel.NLen)
	melTensor.Data = mel.Data

	// Encode
	encoderOut, err := p.encoder.Encode(ctx, melTensor)
	if err != nil {
		return "", ml.Tensor{}, fmt.Errorf("language detection: encoder: %w", err)
	}

	// Create prompt: [SOT, <|lang|>, <|transcribe|>]
	special := p.vocab.Special()
	prompt := []int32{int32(special.SOT)}

	// Decode 1 token to sample language
	decoderParams := DecoderParams{
		Prompt:    prompt,
		MaxTokens: 1,
	}

	segments, err := p.decoder.Decode(ctx, encoderOut, decoderParams)
	if err != nil {
		return "", ml.Tensor{}, fmt.Errorf("language detection: decode: %w", err)
	}

	// Extract language from first token
	if len(segments) > 0 && len(segments[0].Tokens) > 0 {
		tokenID := segments[0].Tokens[0].ID
		langToken := p.vocab.DecodeToken(vocab.Token(tokenID))
		// Extract language code from token like "<|en|>"
		if len(langToken) > 4 && langToken[0] == '<' && langToken[1] == '|' && langToken[len(langToken)-1] == '>' && langToken[len(langToken)-2] == '|' {
			lang := langToken[2 : len(langToken)-2]
			return lang, encoderOut, nil
		}
	}

	// Default to English if detection fails
	return "en", encoderOut, nil
}

// buildPrompt constructs the decoder prompt.
func (p *WhisperPipeline) buildPrompt(language string, params TranscribeParams, isFirstChunk bool) []int32 {
	special := p.vocab.Special()
	prompt := make([]int32, 0, 10)

	// SOT token
	prompt = append(prompt, int32(special.SOT))

	// Language token
	if language != "" {
		if langToken, ok := p.vocab.LanguageID(language); ok {
			prompt = append(prompt, int32(langToken))
		}
	}

	// Task token (transcribe or translate)
	if params.Translate {
		prompt = append(prompt, int32(special.Translate))
	} else {
		prompt = append(prompt, int32(special.Transcribe))
	}

	// No timestamps token if needed
	if params.NoTimestamps {
		prompt = append(prompt, int32(special.NotSOT))
	}

	// Initial prompt (only on first chunk if CarryInitialPrompt is set)
	if isFirstChunk && params.InitialPrompt != "" {
		initialTokens := p.vocab.Encode(params.InitialPrompt)
		for _, t := range initialTokens {
			prompt = append(prompt, int32(t))
		}
	}

	return prompt
}

// postProcessSegments applies post-processing filters to segments.
func (p *WhisperPipeline) postProcessSegments(ctx context.Context, segments []Segment, params TranscribeParams, chunkOffsetMs int64) ([]Segment, error) {
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}

	result := make([]Segment, 0, len(segments))

	for _, seg := range segments {
		// Apply max-len filter
		if params.MaxLen > 0 && len(seg.Tokens) > params.MaxLen {
			if params.SplitOnWord {
				// Split at word boundaries (space tokens)
				seg = p.splitOnWord(seg, params.MaxLen)
			} else {
				seg.Tokens = seg.Tokens[:params.MaxLen]
			}
		}

		// Apply offset/duration filtering
		segStart := seg.StartMs + chunkOffsetMs
		segEnd := seg.EndMs + chunkOffsetMs

		if params.OffsetMs > 0 && segEnd <= int64(params.OffsetMs) {
			continue
		}
		if params.DurationMs > 0 && segStart >= int64(params.OffsetMs+params.DurationMs) {
			continue
		}

		// Adjust segment timestamps
		seg.StartMs = segStart
		seg.EndMs = segEnd

		result = append(result, seg)
	}

	return result, nil
}

// splitOnWord splits a segment at word boundaries if it exceeds maxLen tokens.
func (p *WhisperPipeline) splitOnWord(seg Segment, maxLen int) Segment {
	if len(seg.Tokens) <= maxLen {
		return seg
	}

	// Truncate at word boundary (look for space token)
	spaceTokenID := int32(-1)
	for i := maxLen - 1; i >= 0; i-- {
		if seg.Tokens[i].Text == " " {
			spaceTokenID = seg.Tokens[i].ID
			seg.Tokens = seg.Tokens[:i]
			break
		}
	}

	if spaceTokenID == -1 {
		// No space found, just truncate
		seg.Tokens = seg.Tokens[:maxLen]
	}

	return seg
}

// minInt returns the minimum of two integers.
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
