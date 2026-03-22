package model

import (
	"context"
	"fmt"
	"sync"

	"github.com/whispergo/whisper.go/internal/audio"
)

// WhisperStreamer implements the Streamer interface for real-time transcription.
type WhisperStreamer struct {
	pipeline *WhisperPipeline
	params   TranscribeParams
	ctx      context.Context
	cancel   context.CancelFunc

	audioIn chan []float32
	results chan Segment

	mu          sync.Mutex
	audioBuffer []float32
	err         error
	sampleCount int64
}

// NewStreamer creates a new real-time streamer.
func NewStreamer(ctx context.Context, p *WhisperPipeline, params TranscribeParams) *WhisperStreamer {
	sCtx, sCancel := context.WithCancel(ctx)
	s := &WhisperStreamer{
		pipeline: p,
		params:   params,
		ctx:      sCtx,
		cancel:   sCancel,
		audioIn:  make(chan []float32, 64),
		results:  make(chan Segment, 32),
	}

	if s.params.Logger != nil {
		s.params.Logger.Info("streamer started")
	}

	go s.run()
	return s
}

// Push adds new audio samples to the internal buffer via a channel.
func (s *WhisperStreamer) Push(samples []float32) error {
	select {
	case <-s.ctx.Done():
		return s.ctx.Err()
	case s.audioIn <- samples:
		return nil
	default:
		if s.params.Logger != nil {
			s.params.Logger.Warn("streamer input buffer full, dropping samples")
		}
		return fmt.Errorf("streamer: input buffer full")
	}
}

// Results returns a channel for receiving transcribed segments.
func (s *WhisperStreamer) Results() <-chan Segment {
	return s.results
}

// Close finalizes the stream and releases resources.
func (s *WhisperStreamer) Close() error {
	if s.params.Logger != nil {
		s.params.Logger.Info("streamer closing")
	}
	s.cancel()
	return nil
}

func (s *WhisperStreamer) run() {
	defer close(s.results)

	windowSamples := audio.ChunkSize * audio.SampleRate
	minProcessSamples := 2 * audio.SampleRate

	for {
		select {
		case <-s.ctx.Done():
			return
		case samples := <-s.audioIn:
			s.mu.Lock()
			s.audioBuffer = append(s.audioBuffer, samples...)

			// Only process if we have enough audio
			if len(s.audioBuffer) < minProcessSamples {
				s.mu.Unlock()
				continue
			}

			// Extract chunk
			processLen := minInt(len(s.audioBuffer), windowSamples)
			chunk := make([]float32, processLen)
			copy(chunk, s.audioBuffer[:processLen])

			// Consume audio (simplified rolling window)
			consumeLen := processLen
			if processLen >= windowSamples {
				consumeLen = windowSamples / 2
			}
			s.audioBuffer = s.audioBuffer[consumeLen:]
			currentSampleCount := s.sampleCount
			s.sampleCount += int64(consumeLen)
			s.mu.Unlock()

			if s.params.Logger != nil {
				s.params.Logger.Debug("streamer processing chunk", "samples", len(chunk), "offset_ms", currentSampleCount*1000/int64(audio.SampleRate))
			}

			// Run transcription
			res, err := s.pipeline.Transcribe(s.ctx, chunk, s.params)
			if err != nil {
				if s.params.Logger != nil {
					s.params.Logger.Error("streamer transcription failed", "error", err)
				}
				s.mu.Lock()
				s.err = err
				s.mu.Unlock()
				return
			}

			// Emit results
			offsetMs := currentSampleCount * 1000 / int64(audio.SampleRate)
			for _, seg := range res.Segments {
				seg.StartMs += offsetMs
				seg.EndMs += offsetMs

				if s.params.Logger != nil {
					s.params.Logger.Info("streamer result", "text", seg.Text, "start", seg.StartMs, "end", seg.EndMs)
				}

				select {
				case <-s.ctx.Done():
					return
				case s.results <- seg:
				}
			}
		}
	}
}
