package audio

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// LoadWAV reads a WAV file, converts to mono float32 PCM, and resamples to SampleRate if needed.
func LoadWAV(ctx context.Context, path string) ([]float32, int, error) {
	left, right, origRate, err := loadWAVChannels(ctx, path)
	if err != nil {
		return nil, 0, err
	}

	// Mix stereo to mono by averaging channels.
	mono := make([]float32, len(left))
	for i := range left {
		mono[i] = (left[i] + right[i]) * 0.5
	}

	if origRate == SampleRate {
		return mono, SampleRate, nil
	}

	resampled := resample(mono, origRate, SampleRate)
	return resampled, SampleRate, nil
}

// LoadWAVStereo reads a WAV file and returns both channels as float32.
// If mono, both channels are identical.
func LoadWAVStereo(ctx context.Context, path string) (left, right []float32, err error) {
	l, r, origRate, err := loadWAVChannels(ctx, path)
	if err != nil {
		return nil, nil, err
	}

	if origRate == SampleRate {
		return l, r, nil
	}

	return resample(l, origRate, SampleRate), resample(r, origRate, SampleRate), nil
}

// wavHeader holds parsed WAV format information.
type wavHeader struct {
	audioFormat   uint16
	numChannels   uint16
	sampleRate    uint32
	bitsPerSample uint16
}

// loadWAVChannels parses the WAV file and returns left/right channels at the file's original sample rate.
func loadWAVChannels(ctx context.Context, path string) (left, right []float32, sampleRate int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("audio: wav: open: %w", err)
	}
	defer f.Close()

	hdr, dataSize, err := parseWAVHeader(f)
	if err != nil {
		return nil, nil, 0, err
	}

	bytesPerSample := int(hdr.bitsPerSample) / 8
	frameSize := bytesPerSample * int(hdr.numChannels)
	nFrames := int(dataSize) / frameSize

	left = make([]float32, nFrames)
	if hdr.numChannels > 1 {
		right = make([]float32, nFrames)
	}

	buf := make([]byte, frameSize)
	for i := 0; i < nFrames; i++ {
		if i%4096 == 0 {
			select {
			case <-ctx.Done():
				return nil, nil, 0, fmt.Errorf("audio: wav: read: %w", ctx.Err())
			default:
			}
		}

		if _, err := io.ReadFull(f, buf); err != nil {
			return nil, nil, 0, fmt.Errorf("audio: wav: read sample: %w", err)
		}

		ch0, ch1 := decodeSample(buf, bytesPerSample, hdr.audioFormat)
		left[i] = ch0
		if hdr.numChannels > 1 {
			right[i] = ch1
		}
	}

	if hdr.numChannels == 1 {
		right = make([]float32, nFrames)
		copy(right, left)
	}

	return left, right, int(hdr.sampleRate), nil
}

// decodeSample reads one or two channel samples from buf.
func decodeSample(buf []byte, bytesPerSample int, audioFormat uint16) (ch0, ch1 float32) {
	ch0 = sampleToFloat32(buf[:bytesPerSample], audioFormat)
	if len(buf) > bytesPerSample {
		ch1 = sampleToFloat32(buf[bytesPerSample:bytesPerSample*2], audioFormat)
	}
	return
}

// sampleToFloat32 converts a raw PCM sample to float32 in [-1, 1].
func sampleToFloat32(b []byte, audioFormat uint16) float32 {
	switch len(b) {
	case 1: // 8-bit unsigned PCM
		return (float32(b[0]) - 128.0) / 128.0
	case 2: // 16-bit signed PCM
		v := int16(binary.LittleEndian.Uint16(b))
		return float32(v) / 32768.0
	case 3: // 24-bit signed PCM
		raw := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16
		if raw&0x800000 != 0 {
			raw |= 0xFF000000 // sign-extend
		}
		return float32(int32(raw)) / 8388608.0
	case 4:
		if audioFormat == 3 { // IEEE float
			bits := binary.LittleEndian.Uint32(b)
			return math.Float32frombits(bits)
		}
		// 32-bit signed PCM
		v := int32(binary.LittleEndian.Uint32(b))
		return float32(v) / 2147483648.0
	}
	return 0
}

// parseWAVHeader reads and validates the RIFF/WAVE header and fmt chunk.
// It leaves the file positioned at the start of the PCM data.
func parseWAVHeader(r io.Reader) (wavHeader, uint32, error) {
	// Read RIFF chunk descriptor (12 bytes).
	var riffID [4]byte
	if _, err := io.ReadFull(r, riffID[:]); err != nil {
		return wavHeader{}, 0, fmt.Errorf("audio: wav: read RIFF id: %w", err)
	}
	if string(riffID[:]) != "RIFF" {
		return wavHeader{}, 0, fmt.Errorf("audio: wav: not a RIFF file")
	}

	var chunkSize uint32
	if err := binary.Read(r, binary.LittleEndian, &chunkSize); err != nil {
		return wavHeader{}, 0, fmt.Errorf("audio: wav: read chunk size: %w", err)
	}

	var waveID [4]byte
	if _, err := io.ReadFull(r, waveID[:]); err != nil {
		return wavHeader{}, 0, fmt.Errorf("audio: wav: read WAVE id: %w", err)
	}
	if string(waveID[:]) != "WAVE" {
		return wavHeader{}, 0, fmt.Errorf("audio: wav: not a WAVE file")
	}

	var hdr wavHeader
	var dataSize uint32
	foundFmt := false
	foundData := false

	for !foundData {
		var subID [4]byte
		if _, err := io.ReadFull(r, subID[:]); err != nil {
			return wavHeader{}, 0, fmt.Errorf("audio: wav: read sub-chunk id: %w", err)
		}
		var subSize uint32
		if err := binary.Read(r, binary.LittleEndian, &subSize); err != nil {
			return wavHeader{}, 0, fmt.Errorf("audio: wav: read sub-chunk size: %w", err)
		}

		switch string(subID[:]) {
		case "fmt ":
			if subSize < 16 {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: fmt chunk too small")
			}
			if err := binary.Read(r, binary.LittleEndian, &hdr.audioFormat); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read audioFormat: %w", err)
			}
			if err := binary.Read(r, binary.LittleEndian, &hdr.numChannels); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read numChannels: %w", err)
			}
			if err := binary.Read(r, binary.LittleEndian, &hdr.sampleRate); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read sampleRate: %w", err)
			}
			var byteRate uint32
			if err := binary.Read(r, binary.LittleEndian, &byteRate); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read byteRate: %w", err)
			}
			var blockAlign uint16
			if err := binary.Read(r, binary.LittleEndian, &blockAlign); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read blockAlign: %w", err)
			}
			if err := binary.Read(r, binary.LittleEndian, &hdr.bitsPerSample); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: read bitsPerSample: %w", err)
			}
			if hdr.audioFormat != 1 && hdr.audioFormat != 3 {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: unsupported audio format %d", hdr.audioFormat)
			}
			// Skip extra fmt bytes if present.
			if subSize > 16 {
				if _, err := io.CopyN(io.Discard, r, int64(subSize-16)); err != nil {
					return wavHeader{}, 0, fmt.Errorf("audio: wav: skip fmt extra: %w", err)
				}
			}
			foundFmt = true
		case "data":
			if !foundFmt {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: data chunk before fmt chunk")
			}
			dataSize = subSize
			foundData = true
		default:
			// Skip unknown chunks (LIST, bext, etc.).
			skip := int64(subSize)
			if skip%2 != 0 {
				skip++ // RIFF chunks are word-aligned
			}
			if _, err := io.CopyN(io.Discard, r, skip); err != nil {
				return wavHeader{}, 0, fmt.Errorf("audio: wav: skip chunk: %w", err)
			}
		}
	}

	return hdr, dataSize, nil
}

// resample performs linear interpolation resampling from srcRate to dstRate.
func resample(samples []float32, srcRate, dstRate int) []float32 {
	if srcRate == dstRate {
		return samples
	}
	ratio := float64(srcRate) / float64(dstRate)
	outLen := int(math.Round(float64(len(samples)) * float64(dstRate) / float64(srcRate)))
	out := make([]float32, outLen)
	for i := range out {
		pos := float64(i) * ratio
		idx := int(pos)
		frac := float32(pos - float64(idx))
		if idx+1 < len(samples) {
			out[i] = samples[idx]*(1-frac) + samples[idx+1]*frac
		} else if idx < len(samples) {
			out[i] = samples[idx]
		}
	}
	return out
}
