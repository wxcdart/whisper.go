// Command whisper transcribes audio files to text using a Whisper GGUF model.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/whispergo/whisper.go"
	"github.com/whispergo/whisper.go/internal/audio"
	"github.com/whispergo/whisper.go/internal/model"
	"github.com/whispergo/whisper.go/internal/output"
)

func main() {
	if err := run(context.Background()); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	// Input/Output flags
	var (
		inputFile  string
		outputFile string
		ofTxt      bool
		ofVTT      bool
		ofSRT      bool
		ofJSON     bool
		ofJSONFull bool
		ofCSV      bool
		ofLRC      bool
		ofWTS      bool
	)

	// Model flags
	var modelPath string

	// VAD flags
	var (
		vad      bool
		vadModel string
	)

	// Processing flags
	var (
		threads            int
		language           string
		translate          bool
		splitOnWord        bool
		outputTxt          bool
		maxLen             int
		offsetMs           int
		durationMs         int
		noContext          int
		audioCtx           int
		maxTokens          int
		suppressNST        bool
		suppressRegex      string
		printColors        bool
		printSpecial       bool
		printConfidence    bool
		noTimestamps       bool
		promptLang         string
		initialPrompt      string
		carryInitialPrompt bool
	)

	// Sampling flags
	var (
		temperature    float64
		temperatureInc float64
		threshold      float64
		entropyThold   float64
		logprobThold   float64
	)

	// Beam search flags
	var (
		bestOf   int
		beamSize int
	)

	// Advanced flags
	var (
		modelType          string
		dtwTokenTimestamps bool
		verbose            int
	)

	// Define flags - matching whisper.cpp exactly
	fs := flag.NewFlagSet("whisper", flag.ContinueOnError)
	fs.StringVar(&inputFile, "file", "", "input audio file (WAV, MP3, etc.)")
	fs.StringVar(&inputFile, "f", "", "input audio file (WAV, MP3, etc.) [short form]")

	fs.StringVar(&outputFile, "output-file", "", "output file (if specified, use extension to determine format)")
	fs.StringVar(&outputFile, "of", "", "output file [short form]")
	fs.BoolVar(&ofTxt, "of-txt", false, "write .txt output")
	fs.BoolVar(&ofVTT, "of-vtt", false, "write .vtt output")
	fs.BoolVar(&ofSRT, "of-srt", false, "write .srt output")
	fs.BoolVar(&ofJSON, "of-json", false, "write .json output")
	fs.BoolVar(&ofJSONFull, "of-json-full", false, "write .json (full, with tokens)")
	fs.BoolVar(&ofCSV, "of-csv", false, "write .csv output")
	fs.BoolVar(&ofLRC, "of-lrc", false, "write .lrc output")
	fs.BoolVar(&ofWTS, "of-wts", false, "write .wts output")

	fs.StringVar(&modelPath, "model", "models/ggml-tiny.en.gguf", "path to GGUF model file")
	fs.StringVar(&modelPath, "m", "models/ggml-tiny.en.gguf", "path to GGUF model file [short form]")

	fs.BoolVar(&vad, "vad", false, "enable voice activity detection")
	fs.StringVar(&vadModel, "vad-model", "", "path to Silero VAD GGUF model")

	fs.IntVar(&threads, "threads", 4, "number of threads")
	fs.IntVar(&threads, "t", 4, "number of threads [short form]")
	fs.StringVar(&language, "language", "", "BCP-47 language code (auto-detect if not set)")
	fs.StringVar(&language, "l", "", "BCP-47 language code [short form]")
	fs.BoolVar(&translate, "translate", false, "translate to English")
	fs.BoolVar(&splitOnWord, "split-on-word", false, "split segments on word boundaries")
	fs.BoolVar(&splitOnWord, "su", false, "split segments on word boundaries [short form]")
	fs.BoolVar(&outputTxt, "output-txt", false, "write .txt to same directory as input file")
	fs.BoolVar(&outputTxt, "otxt", false, "write .txt to same directory as input file [short form]")
	fs.IntVar(&maxLen, "max-len", 0, "max tokens per segment (0=unlimited)")
	fs.IntVar(&maxLen, "ml", 0, "max tokens per segment [short form]")
	fs.IntVar(&offsetMs, "offset-t", 0, "start transcription at audio offset (milliseconds)")
	fs.IntVar(&durationMs, "duration", 0, "transcribe for this duration (milliseconds, 0=to end)")
	fs.IntVar(&noContext, "no-context", 0, "disable context (set to 0)")
	fs.IntVar(&noContext, "nc", 0, "disable context [short form]")
	fs.IntVar(&audioCtx, "audio-ctx", 0, "audio context (set to 0 for full audio)")
	fs.IntVar(&maxTokens, "max-tokens", 128, "max tokens per inference")
	fs.BoolVar(&suppressNST, "suppress-non-text-tokens", false, "suppress special tokens in output")
	fs.BoolVar(&suppressNST, "ss", false, "suppress special tokens in output [short form]")
	fs.StringVar(&suppressRegex, "suppress-regex", "", "regex patterns to suppress in output")
	fs.StringVar(&suppressRegex, "sr", "", "regex patterns to suppress in output [short form]")
	fs.BoolVar(&printColors, "print-colors", false, "print with ANSI colors")
	fs.BoolVar(&printColors, "pp", false, "print with ANSI colors [short form]")
	fs.BoolVar(&printSpecial, "print-special", false, "print special tokens")
	fs.BoolVar(&printSpecial, "psp", false, "print special tokens [short form]")
	fs.BoolVar(&printConfidence, "print-confidence", false, "print token confidence scores")
	fs.BoolVar(&printConfidence, "pc", false, "print token confidence scores [short form]")
	fs.BoolVar(&noTimestamps, "no-timestamps", false, "omit timestamps from output")
	fs.BoolVar(&noTimestamps, "nts", false, "omit timestamps from output [short form]")
	fs.StringVar(&promptLang, "prompt-lang", "", "language of initial prompt (same as input language if not set)")
	fs.StringVar(&promptLang, "pml", "", "language of initial prompt [short form]")
	fs.StringVar(&initialPrompt, "initial-prompt", "", "initial prompt text")
	fs.StringVar(&initialPrompt, "ip", "", "initial prompt text [short form]")
	fs.BoolVar(&carryInitialPrompt, "carry-initial-prompt", false, "carry prompt across chunks")
	fs.BoolVar(&carryInitialPrompt, "cip", false, "carry prompt across chunks [short form]")

	fs.Float64Var(&temperature, "temperature", 0.0, "sampling temperature")
	fs.Float64Var(&temperatureInc, "temperature-inc", 0.2, "temperature increase per fallback")
	fs.Float64Var(&threshold, "threshold", 0.1, "no-speech threshold")
	fs.Float64Var(&threshold, "th", 0.1, "no-speech threshold [short form]")
	fs.Float64Var(&entropyThold, "entropy-thold", 2.4, "entropy threshold for fallback")
	fs.Float64Var(&logprobThold, "logprob-thold", -1.0, "logprob threshold for fallback")

	fs.IntVar(&bestOf, "best-of", 5, "number of candidates (for selecting best hypothesis)")
	fs.IntVar(&bestOf, "b", 5, "number of candidates [short form]")
	fs.IntVar(&beamSize, "beam-size", 1, "beam search width (1 = greedy)")
	fs.IntVar(&beamSize, "bs", 1, "beam search width [short form]")

	fs.StringVar(&modelType, "model-type", "", "hint for model type (for debug)")
	fs.BoolVar(&dtwTokenTimestamps, "dtw-token-timestamps", false, "enable DTW alignment")
	fs.BoolVar(&dtwTokenTimestamps, "dtw", false, "enable DTW alignment [short form]")
	fs.IntVar(&verbose, "verbose", 1, "verbosity level (0=quiet, 1=normal, 2=debug)")
	fs.IntVar(&verbose, "v", 1, "verbosity level [short form]")

	if err := fs.Parse(os.Args[1:]); err != nil {
		return err
	}

	// Validate input file
	if inputFile == "" {
		return fmt.Errorf("input file (-f/--file) is required")
	}

	// Load audio
	if verbose > 0 {
		fmt.Fprintf(os.Stderr, "Loading audio from %s\n", inputFile)
	}
	samples, sampleRate, err := audio.LoadWAV(ctx, inputFile)
	if err != nil {
		return fmt.Errorf("failed to load audio: %w", err)
	}

	// Resample if needed
	if sampleRate != audio.SampleRate {
		samples = resample(samples, sampleRate, audio.SampleRate)
		if verbose > 0 {
			fmt.Fprintf(os.Stderr, "Resampled from %d to %d Hz\n", sampleRate, audio.SampleRate)
		}
	}

	// Apply offset and duration
	if offsetMs > 0 || durationMs > 0 {
		offsetSamples := int64(offsetMs) * int64(audio.SampleRate) / 1000
		var endSamples int64 = int64(len(samples))
		if durationMs > 0 {
			endSamples = offsetSamples + int64(durationMs)*int64(audio.SampleRate)/1000
		}
		if offsetSamples < int64(len(samples)) {
			if endSamples > int64(len(samples)) {
				endSamples = int64(len(samples))
			}
			samples = samples[offsetSamples:endSamples]
		}
	}

	// Create transcriber context
	if verbose > 0 {
		fmt.Fprintf(os.Stderr, "Loading model from %s\n", modelPath)
	}

	// Determine task
	task := whisper.TaskTranscribe
	if translate {
		task = whisper.TaskTranslate
	}

	// Build params
	params := whisper.Params{
		Language:           language,
		Task:               task,
		Threads:            threads,
		BeamSize:           beamSize,
		BestOf:             bestOf,
		Temperature:        float32(temperature),
		TemperatureInc:     float32(temperatureInc),
		EntropyThold:       float32(entropyThold),
		LogprobThold:       float32(logprobThold),
		NoSpeechThold:      float32(threshold),
		NoFallback:         false,
		MaxLen:             maxLen,
		SplitOnWord:        splitOnWord,
		NoTimestamps:       noTimestamps,
		InitialPrompt:      initialPrompt,
		CarryInitialPrompt: carryInitialPrompt,
		OffsetMs:           offsetMs,
		DurationMs:         durationMs,
		MaxContext:         0,
		AudioCtx:           audioCtx,
		SuppressNST:        suppressNST,
		SuppressRegex:      suppressRegex,
		VADEnabled:         vad,
		VADModelPath:       vadModel,
	}
	if verbose >= 2 {
		params.Logger = &cliLogger{verbose: verbose}
	}

	// Handle no-context flag compatibility
	if noContext == 0 {
		params.MaxContext = 0
	}

	// Enable DTW if requested
	if dtwTokenTimestamps {
		params.DTWPreset = "default"
	}

	transcriber, err := whisper.New(ctx, modelPath, params)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	// Transcribe
	if verbose > 0 {
		fmt.Fprintf(os.Stderr, "Transcribing audio...\n")
	}
	transcribeStart := time.Now()
	result, err := transcriber.Transcribe(ctx, samples, params)
	if err != nil {
		return fmt.Errorf("transcription failed: %w", err)
	}
	transcribeMs := time.Since(transcribeStart).Milliseconds()

	if verbose > 0 {
		fmt.Fprintf(os.Stderr, "Detected language: %s\n", result.Language)
	}

	// Determine output formats
	formats := determineOutputFormats(outputFile, ofTxt, ofVTT, ofSRT, ofJSON, ofJSONFull, ofCSV, ofLRC, ofWTS)
	if len(formats) == 0 {
		formats = []string{"txt"} // default to txt
	}

	// Write output
	outOpts := output.Options{
		NoTimestamps:    noTimestamps,
		PrintColors:     printColors,
		PrintConfidence: printConfidence,
		PrintSpecial:    printSpecial,
	}

	for _, format := range formats {
		outPath := getOutputPath(inputFile, outputFile, format, outputTxt)
		if verbose > 0 {
			fmt.Fprintf(os.Stderr, "Writing %s output to %s\n", format, outPath)
		}

		if err := writeOutput(ctx, result, format, outPath, outOpts); err != nil {
			return fmt.Errorf("failed to write %s output: %w", format, err)
		}
	}

	if verbose > 0 {
		fmt.Fprintf(os.Stderr, "Transcription complete in %d ms\n", transcribeMs)
	}

	return nil
}

// determineOutputFormats returns the list of output formats based on flags
func determineOutputFormats(outputFile string, ofTxt, ofVTT, ofSRT, ofJSON, ofJSONFull, ofCSV, ofLRC, ofWTS bool) []string {
	var formats []string

	// Explicit format flags take priority
	if ofTxt {
		formats = append(formats, "txt")
	}
	if ofVTT {
		formats = append(formats, "vtt")
	}
	if ofSRT {
		formats = append(formats, "srt")
	}
	if ofJSON {
		formats = append(formats, "json")
	}
	if ofJSONFull {
		formats = append(formats, "json-full")
	}
	if ofCSV {
		formats = append(formats, "csv")
	}
	if ofLRC {
		formats = append(formats, "lrc")
	}
	if ofWTS {
		formats = append(formats, "wts")
	}

	// If explicit output file, infer format from extension
	if outputFile != "" && len(formats) == 0 {
		ext := strings.ToLower(filepath.Ext(outputFile))
		if ext != "" {
			ext = ext[1:] // remove leading dot
			formats = append(formats, ext)
		}
	}

	return formats
}

// getOutputPath returns the path for the output file
func getOutputPath(inputFile, outputFile, format string, outputTxt bool) string {
	if outputFile != "" {
		return outputFile
	}

	if outputTxt {
		return strings.TrimSuffix(inputFile, filepath.Ext(inputFile)) + ".txt"
	}

	return strings.TrimSuffix(inputFile, filepath.Ext(inputFile)) + "." + format
}

// writeOutput writes the result in the specified format to a file
func writeOutput(ctx context.Context, result *whisper.Result, format, path string, opts output.Options) error {
	formatter, err := output.Format(format)
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer f.Close()

	// Cast whisper.Result to model.Result (same memory layout)
	modelResult := (*model.Result)(unsafe.Pointer(result))

	if err := formatter.Format(ctx, f, modelResult, opts); err != nil {
		return err
	}

	return f.Sync()
}

// resample resamples audio from src rate to dst rate using simple linear interpolation
func resample(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate {
		return samples
	}

	ratio := float64(dstRate) / float64(srcRate)
	outLen := int(float64(len(samples)) * ratio)
	out := make([]float32, outLen)

	for i := 0; i < outLen; i++ {
		pos := float64(i) / ratio
		idx := int(pos)
		frac := pos - float64(idx)

		if idx+1 >= len(samples) {
			out[i] = samples[len(samples)-1]
		} else {
			out[i] = samples[idx]*(1-float32(frac)) + samples[idx+1]*float32(frac)
		}
	}

	return out
}

type cliLogger struct {
	verbose int
}

func (l *cliLogger) Debug(msg string, args ...any) {
	if l.verbose < 2 {
		return
	}
	l.log("DEBUG", msg, args...)
}

func (l *cliLogger) Info(msg string, args ...any) {
	if l.verbose < 2 {
		return
	}
	l.log("INFO", msg, args...)
}

func (l *cliLogger) Warn(msg string, args ...any) {
	l.log("WARN", msg, args...)
}

func (l *cliLogger) Error(msg string, args ...any) {
	l.log("ERROR", msg, args...)
}

func (l *cliLogger) log(level, msg string, args ...any) {
	fmt.Fprintf(os.Stderr, "[%s] %s", level, msg)
	for i := 0; i+1 < len(args); i += 2 {
		fmt.Fprintf(os.Stderr, " %v=%v", args[i], args[i+1])
	}
	if len(args)%2 == 1 {
		fmt.Fprintf(os.Stderr, " extra=%v", args[len(args)-1])
	}
	fmt.Fprintf(os.Stderr, "\n")
}
