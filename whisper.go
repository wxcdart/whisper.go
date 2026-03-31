package whisper

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/whispergo/whisper.go/internal/audio"
	"github.com/whispergo/whisper.go/internal/dtw"
	"github.com/whispergo/whisper.go/internal/ggml"
	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/model"
	"github.com/whispergo/whisper.go/internal/vad"
	"github.com/whispergo/whisper.go/internal/vocab"
)

// SampleRate is the required input audio sample rate.
const SampleRate = 16000

// Context holds a loaded model and is safe for concurrent use across multiple Transcribe calls.
type Context struct {
	enc        model.Encoder
	dec        model.Decoder
	vocab      *vocab.Vocabulary
	pipeline   *model.WhisperPipeline
	dtwAligner *dtw.Aligner
	vadModel   *vad.SileroVAD
	file       gguf.FileLike
}

// Params controls transcription behaviour.
type Params struct {
	Language           string // BCP-47 language code, empty = auto-detect
	Task               Task
	Threads            int
	Processors         int
	BeamSize           int
	BestOf             int
	MaxTokens          int
	Temperature        float32
	TemperatureInc     float32
	EntropyThold       float32
	LogprobThold       float32
	NoSpeechThold      float32
	NoFallback         bool
	MaxLen             int
	SplitOnWord        bool
	NoTimestamps       bool
	InitialPrompt      string
	CarryInitialPrompt bool
	OffsetMs           int
	DurationMs         int
	MaxContext         int
	AudioCtx           int
	SuppressNST        bool
	SuppressRegex      string
	DTWPreset          string
	Logger             model.Logger // Optional logger for progress/timing
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
		MaxTokens:         128,
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
	return NewWithModelBackend(ctx, modelPath, nil, params...)
}

// NewWithModelBackend loads a model from the given GGUF file path and configures encoder/decoder
// to use the provided model compute backend. If backend is nil, the default ml backend is used.
func NewWithModelBackend(ctx context.Context, modelPath string, backend model.ComputeBackend, params ...Params) (*Context, error) {
	// Load GGUF or legacy ggml file
	f, err := ggml.Open(ctx, modelPath)
	if err != nil {
		return nil, fmt.Errorf("load GGUF: %w", err)
	}

	// Load encoder
	enc, err := model.NewEncoderWithBackend(f, backend)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load encoder: %w", err)
	}

	// Load decoder
	dec, err := model.NewDecoderWithBackend(f, backend)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load decoder: %w", err)
	}

	// Load vocabulary
	v, err := loadVocabFromGGUF(f, modelPath)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("load vocab: %w", err)
	}
	dec.SetVocabulary(v)

	// Create pipeline
	pipeline, err := model.New(enc, dec, v, nil, nil)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("create pipeline: %w", err)
	}

	if melFilters, err := loadMelFiltersFromModel(ctx, f); err == nil {
		pipeline.SetMelFilters(melFilters)
	}

	return &Context{
		enc:      enc,
		dec:      dec,
		vocab:    v,
		pipeline: pipeline,
		file:     f,
	}, nil
}

func loadMelFiltersFromModel(ctx context.Context, f gguf.FileLike) (audio.MelFilters, error) {
	keys := []string{
		"mel_filters",
		"model.mel_filters",
		"whisper.mel_filters",
	}

	for _, key := range keys {
		data, shape, err := f.Tensor(ctx, key)
		if err != nil {
			continue
		}
		if len(shape) != 2 {
			continue
		}
		rows, cols := shape[0], shape[1]
		if rows <= 0 || cols <= 0 {
			continue
		}
		if len(data) != rows*cols {
			continue
		}

		// Expected orientation is [n_mel, n_freq] where n_freq = 201.
		nMel := rows
		nFreq := cols
		if rows == audio.NFFT/2+1 && cols == audio.NMel {
			// Transpose [n_freq, n_mel] into [n_mel, n_freq].
			nMel = cols
			nFreq = rows
			transposed := make([]float32, len(data))
			for r := 0; r < rows; r++ {
				for c := 0; c < cols; c++ {
					transposed[c*nFreq+r] = data[r*cols+c]
				}
			}
			return audio.MelFilters{NMel: nMel, Data: transposed}, nil
		}

		return audio.MelFilters{NMel: nMel, Data: data}, nil
	}

	return audio.MelFilters{}, fmt.Errorf("mel filters tensor not found")
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
			MaxTokens:      params.MaxTokens,
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
		Logger:             params.Logger, // Pass through logger
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
func loadVocabFromGGUF(f gguf.FileLike, modelPath string) (*vocab.Vocabulary, error) {
	getMetaAny := func(keys ...string) (any, bool) {
		for _, k := range keys {
			if v, ok := f.Meta(k); ok {
				return v, true
			}
		}
		return nil, false
	}

	// Get tokens metadata from known key variants.
	tokensRaw, ok := getMetaAny(
		"tokenizer.ggml.tokens",
		"tokenizer.tokens",
	)
	if !ok {
		expectedVocab := 0
		if v, ok := f.MetaUint32("whisper.vocab.size"); ok {
			expectedVocab = int(v)
		} else if v, ok := f.MetaUint32("whisper.n_vocab"); ok {
			expectedVocab = int(v)
		}
		return loadVocabFromTokenizerFile(modelPath, expectedVocab)
	}

	toStrings := func(v any) ([]string, bool) {
		s, ok := v.([]string)
		if ok {
			return s, true
		}
		items, ok := v.([]any)
		if !ok {
			return nil, false
		}
		out := make([]string, len(items))
		for i, it := range items {
			ts, ok := it.(string)
			if !ok {
				return nil, false
			}
			out[i] = ts
		}
		return out, true
	}

	tokens, ok := toStrings(tokensRaw)
	if !ok {
		return nil, fmt.Errorf("tokenizer tokens metadata has unsupported type %T", tokensRaw)
	}

	// Get token types metadata from known key variants.
	typesRaw, ok := getMetaAny(
		"tokenizer.ggml.token_type",
		"tokenizer.token_type",
	)
	if !ok {
		return nil, fmt.Errorf("token type metadata not found in GGUF (tried tokenizer.ggml.token_type/tokenizer.token_type)")
	}

	toUint32s := func(v any) ([]uint32, bool) {
		u, ok := v.([]uint32)
		if ok {
			return u, true
		}
		items, ok := v.([]any)
		if !ok {
			return nil, false
		}
		out := make([]uint32, len(items))
		for i, it := range items {
			tv, ok := it.(uint32)
			if !ok {
				return nil, false
			}
			out[i] = tv
		}
		return out, true
	}

	types, ok := toUint32s(typesRaw)
	if !ok {
		return nil, fmt.Errorf("tokenizer token_type metadata has unsupported type %T", typesRaw)
	}

	// Create vocabulary
	return vocab.New(tokens, types)
}

var timestampTokenRe = regexp.MustCompile(`^<\|\d+\.\d+\|>$`)

type hfAddedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

type hfTokenizerFile struct {
	AddedTokens []hfAddedToken `json:"added_tokens"`
	Model       struct {
		Vocab map[string]int `json:"vocab"`
	} `json:"model"`
}

func loadVocabFromTokenizerFile(modelPath string, expectedVocab int) (*vocab.Vocabulary, error) {
	tokPath, err := findTokenizerFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("tokenizer metadata not found in GGUF and sidecar tokenizer file not found near %q", modelPath)
	}

	buf, err := os.ReadFile(tokPath)
	if err != nil {
		return nil, fmt.Errorf("read sidecar tokenizer %q: %w", tokPath, err)
	}

	var tok hfTokenizerFile
	if err := json.Unmarshal(buf, &tok); err != nil {
		return nil, fmt.Errorf("parse sidecar tokenizer %q: %w", tokPath, err)
	}

	maxID := -1
	for _, id := range tok.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, t := range tok.AddedTokens {
		if t.ID > maxID {
			maxID = t.ID
		}
	}
	if expectedVocab > 0 && expectedVocab-1 > maxID {
		maxID = expectedVocab - 1
	}
	if maxID < 0 {
		return nil, fmt.Errorf("sidecar tokenizer %q has no vocabulary entries", tokPath)
	}

	tokens := make([]string, maxID+1)
	types := make([]uint32, maxID+1)

	for token, id := range tok.Model.Vocab {
		if id >= 0 && id < len(tokens) {
			tokens[id] = token
			types[id] = 0
		}
	}
	for _, t := range tok.AddedTokens {
		if t.ID >= 0 && t.ID < len(tokens) {
			tokens[t.ID] = t.Content
			types[t.ID] = 1
		}
	}

	for i := range tokens {
		if tokens[i] == "" {
			tokens[i] = fmt.Sprintf("<|missing_%d|>", i)
		}
		if timestampTokenRe.MatchString(tokens[i]) {
			types[i] = 3
		}
		if strings.HasPrefix(tokens[i], "<|") && strings.HasSuffix(tokens[i], "|>") && types[i] == 0 {
			types[i] = 1
		}
	}

	return vocab.New(tokens, types)
}

func findTokenizerFile(modelPath string) (string, error) {
	dir := filepath.Dir(modelPath)
	modelBase := strings.ToLower(filepath.Base(modelPath))

	preferred := []string{}
	if strings.Contains(modelBase, "tiny") {
		preferred = append(preferred, "tokenizer-tiny.json")
	}
	preferred = append(preferred, "tokenizer.json")

	for _, name := range preferred {
		p := filepath.Join(dir, name)
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}

	ents, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	var matches []string
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, "tokenizer") && strings.HasSuffix(name, ".json") {
			matches = append(matches, filepath.Join(dir, name))
		}
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("not found")
	}
	sort.Strings(matches)
	return matches[0], nil
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
