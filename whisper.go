package whisper

import (
	"context"
	"fmt"

	"github.com/whispergo/whisper.go/internal/dtw"
	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/model"
	"github.com/whispergo/whisper.go/internal/vad"
	"github.com/whispergo/whisper.go/internal/vocab"
)

// SampleRate is the required input audio sample rate.
const SampleRate = 16000

// Context holds a loaded model and is safe for concurrent use across multiple Transcribe calls.
type Context struct {
	enc       model.Encoder
	dec       model.Decoder
	vocab     *vocab.Vocabulary
	pipeline  *model.WhisperPipeline
	dtwAligner *dtw.Aligner
	vadModel  *vad.SileroVAD
	file      *gguf.File
}

// Params controls transcription behaviour.
type Params struct {
	Language        string // BCP-47 language code, empty = auto-detect
	Task            Task
	Threads         int
	Processors      int
	BeamSize        int
	BestOf          int
	Temperature     float32
	TemperatureInc  float32
	EntropyThold    float32
	LogprobThold    float32
	NoSpeechThold   float32
	NoFallback      bool
	MaxLen          int
	SplitOnWord     bool
	NoTimestamps    bool
	InitialPrompt   string
	CarryInitialPrompt bool
	OffsetMs        int
	DurationMs      int
	MaxContext      int
	AudioCtx        int
	SuppressNST     bool
	SuppressRegex   string
	DTWPreset       string
	// VAD
	VADEnabled        bool
	VADModelPath      string
	VADThreshold      float32
	VADMinSpeechMs    int
	VADMinSilenceMs   int
	VADMaxSpeechS     float32
	VADSpeechPadMs    int
	VADSamplesOverlap float32
	// Grammar
	Grammar        string
	GrammarRule    string
	GrammarPenalty float32
	// Callbacks
	NewSegment func(ctx context.Context, result *Result, segIdx int) error
	Progress   func(ctx context.Context, progress int) error
}

// Task is the transcription task type.
type Task int

const (
	TaskTranscribe Task = iota
	TaskTranslate
)

// DefaultParams returns sensible defaults matching whisper.cpp defaults.
func DefaultParams() Params {
	return Params{
		Language:          "en",
		Task:              TaskTranscribe,
		Threads:           4,
		BeamSize:          5,
		BestOf:            5,
		Temperature:       0.0,
		TemperatureInc:    0.2,
		EntropyThold:      2.40,
		LogprobThold:      -1.00,
		NoSpeechThold:     0.6,
		VADThreshold:      0.5,
		VADMinSpeechMs:    250,
		VADMinSilenceMs:   100,
		VADSpeechPadMs:    30,
		VADSamplesOverlap: 0.1,
	}
}

// Result is the full transcription output.
type Result struct {
	Segments []Segment
	Language string
}

// Segment is one transcribed segment with timestamps.
type Segment struct {
	StartMs int64
	EndMs   int64
	Text    string
	Tokens  []TokenData
	Speaker int // -1 if unknown; used by diarize
}

// TokenData holds per-token metadata.
type TokenData struct {
	ID    int32
	Text  string
	P     float32 // probability
	PLog  float32 // log probability
	PT    float32 // timestamp probability
	PTSum float32
	T0    int64 // start ms
	T1    int64 // end ms
	TDTW  int64 // DTW-aligned time ms
}

// New loads a model from the given GGUF file path.
func New(ctx context.Context, modelPath string, params ...Params) (*Context, error) {
	// Load GGUF file
	f, err := gguf.Open(ctx, modelPath)
	if err != nil {
		return nil, fmt.Errorf("load GGUF: %w", err)
	}

	// Load encoder
	enc, err := model.NewEncoder(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load encoder: %w", err)
	}

	// Load decoder
	dec, err := model.NewDecoder(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load decoder: %w", err)
	}

	// Load vocabulary
	v, err := loadVocabFromGGUF(f)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load vocab: %w", err)
	}

	// Create pipeline
	pipeline, err := model.New(enc, dec, v, nil, nil)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("create pipeline: %w", err)
	}

	return &Context{
		enc:      enc,
		dec:      dec,
		vocab:    v,
		pipeline: pipeline,
		file:     f,
	}, nil
}

// Transcribe transcribes audio samples (mono float32 at 16 kHz) and returns the result.
func (c *Context) Transcribe(ctx context.Context, samples []float32, params Params) (*Result, error) {
	// Convert Params to model.TranscribeParams
	modelParams := model.TranscribeParams{
		DecoderParams: model.DecoderParams{
			BeamSize:       params.BeamSize,
			BestOf:         params.BestOf,
			Temperature:    params.Temperature,
			TemperatureInc: params.TemperatureInc,
			EntropyThold:   params.EntropyThold,
			LogprobThold:   params.LogprobThold,
			NoSpeechThold:  params.NoSpeechThold,
			NoFallback:     params.NoFallback,
			MaxTokens:      params.MaxLen,
			SuppressNST:    params.SuppressNST,
			SuppressRegex:  params.SuppressRegex,
			DTWEnabled:     params.DTWPreset != "",
		},
		Language:           params.Language,
		AutoDetectLanguage: params.Language == "",
		Translate:          params.Task == TaskTranslate,
		MaxLen:             params.MaxLen,
		SplitOnWord:        params.SplitOnWord,
		NoTimestamps:       params.NoTimestamps,
		InitialPrompt:      params.InitialPrompt,
		CarryInitialPrompt: params.CarryInitialPrompt,
		OffsetMs:           params.OffsetMs,
		DurationMs:         params.DurationMs,
		MaxContext:         params.MaxContext,
		AudioCtx:           params.AudioCtx,
		Threads:            params.Threads,
		VADEnabled:         params.VADEnabled,
		VADModelPath:       params.VADModelPath,
	}

	// Run pipeline
	result, err := c.pipeline.Transcribe(ctx, samples, modelParams)
	if err != nil {
		return nil, err
	}

	// Convert internal result to public result
	return &Result{
		Segments: convertSegments(result.Segments),
		Language: result.Language,
	}, nil
}

// Close releases all resources held by the context.
func (c *Context) Close() error {
	if c.file != nil {
		return c.file.Close()
	}
	return nil
}

// loadVocabFromGGUF extracts tokens and token types from GGUF metadata.
func loadVocabFromGGUF(f *gguf.File) (*vocab.Vocabulary, error) {
	// Get tokens metadata
	tokensRaw, ok := f.Meta("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found in GGUF")
	}

	tokens, ok := tokensRaw.([]string)
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not []string")
	}

	// Get token types metadata
	typesRaw, ok := f.Meta("tokenizer.ggml.token_type")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.token_type not found in GGUF")
	}

	types, ok := typesRaw.([]uint32)
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.token_type is not []uint32")
	}

	// Create vocabulary
	return vocab.New(tokens, types)
}

// convertSegments converts internal segments to public segments.
func convertSegments(internal []model.Segment) []Segment {
	out := make([]Segment, len(internal))
	for i, seg := range internal {
		out[i] = Segment{
			StartMs: seg.StartMs,
			EndMs:   seg.EndMs,
			Text:    seg.Text,
			Tokens:  convertTokenData(seg.Tokens),
			Speaker: seg.Speaker,
		}
	}
	return out
}

// convertTokenData converts internal token data to public token data.
func convertTokenData(internal []model.TokenData) []TokenData {
	out := make([]TokenData, len(internal))
	for i, tok := range internal {
		out[i] = TokenData{
			ID:    tok.ID,
			Text:  tok.Text,
			P:     tok.P,
			PLog:  tok.PLog,
			PT:    tok.PT,
			PTSum: tok.PTSum,
			T0:    tok.T0,
			T1:    tok.T1,
			TDTW:  tok.TDTW,
		}
	}
	return out
}
