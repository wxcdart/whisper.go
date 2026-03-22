package vad

import (
	"context"
	"fmt"
	"math"

	"github.com/whispergo/whisper.go/internal/audio"
	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/ml"
)

const (
	// VAD-specific STFT parameters (Silero model expects 16kHz audio)
	VADSampleRate = 16000
	VADFFTSize    = 512
	VADHopLength  = 160 // 10ms stride at 16kHz
	VADMelBins    = 64
)

// SileroVAD implements voice activity detection using the Silero model.
type SileroVAD struct {
	// Encoder layers: 4 CNN layers
	encoderWeights []ml.Tensor // [layer][outC, inC, kernel]
	encoderBiases  []ml.Tensor // [layer][outC]

	// LSTM parameters
	lstmInputWeight  ml.Tensor // [4*hidden, input] for 4 gates (input, forget, cell, output)
	lstmHiddenWeight ml.Tensor // [4*hidden, hidden]
	lstmBias         ml.Tensor // [4*hidden]
	lstmHidden       int       // hidden dimension (64)

	// Classifier: Conv1D(64 -> 1, kernel=1) + Sigmoid
	classifierWeight ml.Tensor // [1, 64, 1]
	classifierBias   ml.Tensor // [1]

	// Mel filterbank for spectrogram computation
	melFilters audio.MelFilters
}

// lstmState holds the internal state of the LSTM.
type lstmState struct {
	hidden ml.Tensor // [hidden]
	cell   ml.Tensor // [hidden]
}

// NewVAD loads a Silero VAD model from a GGUF file.
func NewVAD(ctx context.Context, f gguf.FileLike) (*SileroVAD, error) {
	vad := &SileroVAD{}

	// Load hyperparameters from metadata
	layerCount, ok := f.MetaUint32("vad.encoder.layer_count")
	if !ok {
		return nil, fmt.Errorf("vad: missing metadata: vad.encoder.layer_count")
	}

	lstmInput, ok := f.MetaUint32("vad.lstm.input_count")
	if !ok {
		return nil, fmt.Errorf("vad: missing metadata: vad.lstm.input_count")
	}

	lstmHidden, ok := f.MetaUint32("vad.lstm.hidden_count")
	if !ok {
		return nil, fmt.Errorf("vad: missing metadata: vad.lstm.hidden_count")
	}

	vad.lstmHidden = int(lstmHidden)

	// Load encoder weights and biases
	vad.encoderWeights = make([]ml.Tensor, layerCount)
	vad.encoderBiases = make([]ml.Tensor, layerCount)

	for i := uint32(0); i < layerCount; i++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("vad: context cancelled: %w", err)
		}

		wKey := fmt.Sprintf("vad.encoder.layer.%d.weight", i)
		bKey := fmt.Sprintf("vad.encoder.layer.%d.bias", i)

		wData, wShape, err := f.Tensor(ctx, wKey)
		if err != nil {
			return nil, fmt.Errorf("vad: load encoder weight %d: %w", i, err)
		}
		vad.encoderWeights[i] = ml.From(wData, wShape...)

		bData, bShape, err := f.Tensor(ctx, bKey)
		if err != nil {
			return nil, fmt.Errorf("vad: load encoder bias %d: %w", i, err)
		}
		vad.encoderBiases[i] = ml.From(bData, bShape...)
	}

	// Load LSTM weights
	lstmInData, _, err := f.Tensor(ctx, "vad.lstm.weights.input")
	if err != nil {
		return nil, fmt.Errorf("vad: load lstm input weights: %w", err)
	}
	vad.lstmInputWeight = ml.From(lstmInData, int(4*lstmHidden), int(lstmInput))

	lstmHidData, _, err := f.Tensor(ctx, "vad.lstm.weights.hidden")
	if err != nil {
		return nil, fmt.Errorf("vad: load lstm hidden weights: %w", err)
	}
	vad.lstmHiddenWeight = ml.From(lstmHidData, int(4*lstmHidden), int(lstmHidden))

	lstmBData, _, err := f.Tensor(ctx, "vad.lstm.bias")
	if err != nil {
		return nil, fmt.Errorf("vad: load lstm bias: %w", err)
	}
	vad.lstmBias = ml.From(lstmBData, int(4*lstmHidden))

	// Load classifier
	clsWData, clsWShape, err := f.Tensor(ctx, "vad.classifier.weight")
	if err != nil {
		return nil, fmt.Errorf("vad: load classifier weight: %w", err)
	}
	vad.classifierWeight = ml.From(clsWData, clsWShape...)

	clsBData, clsBShape, err := f.Tensor(ctx, "vad.classifier.bias")
	if err != nil {
		return nil, fmt.Errorf("vad: load classifier bias: %w", err)
	}
	vad.classifierBias = ml.From(clsBData, clsBShape...)

	// Build mel filterbank (VAD uses 64 mel bins, 80-400 Hz for speech)
	vad.melFilters = buildMelFilters(VADMelBins)

	return vad, nil
}

// Detect performs voice activity detection on audio samples.
func (v *SileroVAD) Detect(ctx context.Context, samples []float32, sampleRate int, params Params) ([]SpeechSegment, error) {
	// Resample if necessary (VAD model expects 16kHz)
	if sampleRate != VADSampleRate {
		samples = resample(samples, sampleRate, VADSampleRate)
	}

	// Compute mel-spectrogram with VAD-specific parameters
	mel, err := computeMelSpectrogram(ctx, samples, v.melFilters)
	if err != nil {
		return nil, fmt.Errorf("vad: mel spectrogram: %w", err)
	}

	// Run through encoder (4 CNN layers with ReLU)
	encoded := ml.From(mel.Data, mel.NMel, mel.NLen)
	for i, w := range v.encoderWeights {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("vad: context cancelled: %w", err)
		}
		encoded, err = ml.Conv1D(ctx, encoded, w, v.encoderBiases[i], 1)
		if err != nil {
			return nil, fmt.Errorf("vad: encoder layer %d: %w", i, err)
		}
		encoded = relu(encoded)
	}

	// Run through LSTM (unidirectional, forward pass)
	lstmOut, err := v.runLSTM(ctx, encoded)
	if err != nil {
		return nil, fmt.Errorf("vad: lstm: %w", err)
	}

	// Classifier: Conv1D(64 -> 1, kernel=1) + Sigmoid
	confidences, err := ml.Conv1D(ctx, lstmOut, v.classifierWeight, v.classifierBias, 1)
	if err != nil {
		return nil, fmt.Errorf("vad: classifier: %w", err)
	}
	confidences = sigmoid(confidences)

	// Extract confidence scores (should be [1, nFrames])
	if len(confidences.Shape) != 2 || confidences.Shape[0] != 1 {
		return nil, fmt.Errorf("vad: unexpected classifier output shape: %v", confidences.Shape)
	}

	scores := confidences.Data // [1 * nFrames] = [nFrames]

	// Apply threshold and post-processing
	segments := v.postProcess(scores, params)

	return segments, nil
}

// runLSTM executes the LSTM forward pass on the encoder output.
func (v *SileroVAD) runLSTM(ctx context.Context, input ml.Tensor) (ml.Tensor, error) {
	// Input shape: [64, nFrames] (encoder output)
	if len(input.Shape) != 2 || input.Shape[0] != v.lstmHidden {
		return ml.Tensor{}, fmt.Errorf("lstm input shape mismatch: got %v, expected [%d, ...]", input.Shape, v.lstmHidden)
	}

	nFrames := input.Shape[1]
	inC := input.Shape[0]

	// Initialize hidden and cell states
	state := lstmState{
		hidden: ml.New(v.lstmHidden),
		cell:   ml.New(v.lstmHidden),
	}

	// Output accumulator: [hidden, nFrames]
	output := ml.New(v.lstmHidden, nFrames)

	// Process each frame sequentially
	for t := 0; t < nFrames; t++ {
		if err := ctx.Err(); err != nil {
			return ml.Tensor{}, fmt.Errorf("lstm: context cancelled: %w", err)
		}

		// Extract input at frame t: [inC]
		xt := ml.New(inC)
		for i := 0; i < inC; i++ {
			xt.Data[i] = input.Data[i*nFrames+t]
		}

		// Compute gates: [4*hidden] = (input @ W_in + hidden @ W_hidden + bias)
		// input @ W_in: [inC] @ [4*hidden, inC]^T = [4*hidden]
		inGates := matmul1D(xt.Data, v.lstmInputWeight, inC, 4*v.lstmHidden)

		// hidden @ W_hidden: [hidden] @ [4*hidden, hidden]^T = [4*hidden]
		hidGates := matmul1D(state.hidden.Data, v.lstmHiddenWeight, v.lstmHidden, 4*v.lstmHidden)

		// Add and apply bias
		gates := make([]float32, 4*v.lstmHidden)
		for i := range gates {
			gates[i] = inGates[i] + hidGates[i] + v.lstmBias.Data[i]
		}

		// Split gates: input, forget, cell, output
		inputGate := sigmoid32(gates[:v.lstmHidden])
		forgetGate := sigmoid32(gates[v.lstmHidden : 2*v.lstmHidden])
		cellGate := tanh32(gates[2*v.lstmHidden : 3*v.lstmHidden])
		outputGate := sigmoid32(gates[3*v.lstmHidden : 4*v.lstmHidden])

		// Update cell state: c_t = f_t * c_{t-1} + i_t * tilde_c_t
		newCell := make([]float32, v.lstmHidden)
		for i := 0; i < v.lstmHidden; i++ {
			newCell[i] = forgetGate[i]*state.cell.Data[i] + inputGate[i]*cellGate[i]
		}

		// Update hidden state: h_t = o_t * tanh(c_t)
		newHidden := make([]float32, v.lstmHidden)
		for i := 0; i < v.lstmHidden; i++ {
			newHidden[i] = outputGate[i] * float32(math.Tanh(float64(newCell[i])))
		}

		// Store output
		copy(output.Data[t*v.lstmHidden:(t+1)*v.lstmHidden], newHidden)

		state.hidden.Data = newHidden
		state.cell.Data = newCell
	}

	return output, nil
}

// postProcess applies thresholding, padding, and merging to segments.
func (v *SileroVAD) postProcess(scores []float32, params Params) []SpeechSegment {
	nFrames := len(scores)
	frameMs := 10 // VAD stride is 10ms

	// Threshold
	segments := make([]SpeechSegment, 0)
	inSegment := false
	var segStart int

	for t := 0; t < nFrames; t++ {
		if scores[t] >= params.Threshold {
			if !inSegment {
				segStart = t
				inSegment = true
			}
		} else {
			if inSegment {
				segEnd := t - 1
				segments = append(segments, SpeechSegment{
					StartMs: int64(segStart * frameMs),
					EndMs:   int64((segEnd + 1) * frameMs),
				})
				inSegment = false
			}
		}
	}
	if inSegment {
		segments = append(segments, SpeechSegment{
			StartMs: int64(segStart * frameMs),
			EndMs:   int64(nFrames * frameMs),
		})
	}

	// Pad segments
	padMs := int64(params.SpeechPadMs)
	for i := range segments {
		segments[i].StartMs = max(0, segments[i].StartMs-padMs)
		segments[i].EndMs = segments[i].EndMs + padMs
	}

	// Merge nearby segments
	merged := make([]SpeechSegment, 0)
	for _, seg := range segments {
		if len(merged) > 0 && seg.StartMs-merged[len(merged)-1].EndMs <= int64(params.MinSilenceMs) {
			merged[len(merged)-1].EndMs = seg.EndMs
		} else {
			merged = append(merged, seg)
		}
	}

	// Filter by minimum duration
	minMs := int64(params.MinSpeechMs)
	filtered := make([]SpeechSegment, 0)
	for _, seg := range merged {
		if seg.EndMs-seg.StartMs >= minMs {
			filtered = append(filtered, seg)
		}
	}

	return filtered
}

// Helper functions

// relu applies ReLU activation element-wise.
func relu(t ml.Tensor) ml.Tensor {
	out := ml.New(t.Shape...)
	for i, v := range t.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}
	return out
}

// sigmoid applies sigmoid activation element-wise.
func sigmoid(t ml.Tensor) ml.Tensor {
	out := ml.New(t.Shape...)
	for i, v := range t.Data {
		out.Data[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
	return out
}

// sigmoid32 applies sigmoid to a float32 slice.
func sigmoid32(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(1 / (1 + math.Exp(-float64(x))))
	}
	return out
}

// tanh32 applies tanh to a float32 slice.
func tanh32(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(math.Tanh(float64(x)))
	}
	return out
}

// matmul1D computes a 1D matrix-vector product: out[i] = sum_j(x[j] * W[i, j])
// W is [outDim, inDim] in row-major order
func matmul1D(x []float32, W ml.Tensor, inDim, outDim int) []float32 {
	out := make([]float32, outDim)
	for i := 0; i < outDim; i++ {
		sum := float32(0)
		for j := 0; j < inDim; j++ {
			sum += x[j] * W.Data[i*inDim+j]
		}
		out[i] = sum
	}
	return out
}

// resample resamples audio from one sample rate to another using linear interpolation.
func resample(samples []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return samples
	}
	ratio := float64(toRate) / float64(fromRate)
	outLen := int(float64(len(samples)) * ratio)
	out := make([]float32, outLen)
	for i := 0; i < outLen; i++ {
		srcPos := float64(i) / ratio
		srcIdx := int(srcPos)
		frac := srcPos - float64(srcIdx)
		if srcIdx+1 < len(samples) {
			out[i] = samples[srcIdx]*(1-float32(frac)) + samples[srcIdx+1]*float32(frac)
		} else if srcIdx < len(samples) {
			out[i] = samples[srcIdx]
		}
	}
	return out
}

// computeMelSpectrogram computes a mel-spectrogram with VAD-specific parameters.
func computeMelSpectrogram(ctx context.Context, samples []float32, filters audio.MelFilters) (audio.Mel, error) {
	// Use STFT parameters: 512 FFT, 160 hop, 64 mel bins
	// We'll compute via FFT directly for VAD's specific window/hop
	nFFT := VADFFTSize
	hopLength := VADHopLength
	nMel := filters.NMel
	nFreq := nFFT/2 + 1

	// Pad samples
	padded := make([]float32, len(samples)+nFFT)
	copy(padded, samples)

	// Hann window
	hann := hannWindow(nFFT)

	// Number of frames
	nFrames := 1 + (len(padded)-nFFT)/hopLength

	// Allocate output
	data := make([]float32, nMel*nFrames)

	// Compute FFT for each frame
	for t := 0; t < nFrames; t++ {
		if err := ctx.Err(); err != nil {
			return audio.Mel{}, fmt.Errorf("mel spectrogram: context cancelled: %w", err)
		}

		pos := t * hopLength

		// Apply window
		windowed := make([]float64, nFFT)
		for i := 0; i < nFFT; i++ {
			if pos+i < len(padded) {
				windowed[i] = float64(padded[pos+i]) * hann[i]
			}
		}

		// FFT (using audio.fftInPlace logic)
		fftSize := nextPow2(nFFT)
		fftBuf := make([]complex128, fftSize)
		for i := 0; i < nFFT; i++ {
			fftBuf[i] = complex(windowed[i], 0)
		}
		fftInPlace(fftBuf)

		// Power spectrum
		power := make([]float64, nFreq)
		for k := 0; k < nFreq; k++ {
			re := real(fftBuf[k])
			im := imag(fftBuf[k])
			power[k] = re*re + im*im
		}

		// Mel filterbank
		for m := 0; m < nMel; m++ {
			sum := 0.0
			for k := 0; k < nFreq; k++ {
				sum += float64(filters.Data[m*nFreq+k]) * power[k]
			}
			if sum < 1e-10 {
				sum = 1e-10
			}
			data[m*nFrames+t] = float32(math.Log10(sum))
		}
	}

	// Normalize
	maxVal := float32(math.Inf(-1))
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	for i := range data {
		v := data[i]
		if v < maxVal-8.0 {
			v = maxVal - 8.0
		}
		data[i] = (v + 4.0) / 4.0
	}

	return audio.Mel{NMel: nMel, NLen: nFrames, Data: data}, nil
}

// buildMelFilters constructs a mel filterbank.
func buildMelFilters(nMel int) audio.MelFilters {
	nFFT := VADFFTSize
	sampleRate := VADSampleRate
	nFreq := nFFT/2 + 1

	// Mel frequency scale
	fMin, fMax := 80.0, 400.0

	// Convert Hz to mel
	melMin := 2595 * math.Log10(1+fMin/700)
	melMax := 2595 * math.Log10(1+fMax/700)

	// Mel points (nMel+2 edges)
	melPts := make([]float64, nMel+2)
	for i := 0; i < nMel+2; i++ {
		melPts[i] = melMin + float64(i)*(melMax-melMin)/float64(nMel+1)
	}

	// Convert mel back to Hz and then to FFT bins
	hzPts := make([]float64, len(melPts))
	binPts := make([]int, len(melPts))
	for i, mel := range melPts {
		hz := 700 * (math.Pow(10, mel/2595) - 1)
		hzPts[i] = hz
		binPts[i] = int(math.Round(hz * float64(nFFT) / float64(sampleRate)))
	}

	// Build filterbank
	filterData := make([]float32, nMel*nFreq)
	for m := 0; m < nMel; m++ {
		left := binPts[m]
		center := binPts[m+1]
		right := binPts[m+2]

		for k := 0; k < nFreq; k++ {
			if k <= left || k >= right {
				continue
			}
			var val float64
			if k <= center {
				val = float64(k-left) / float64(center-left)
			} else {
				val = float64(right-k) / float64(right-center)
			}
			filterData[m*nFreq+k] = float32(val)
		}
	}

	return audio.MelFilters{NMel: nMel, Data: filterData}
}

// hannWindow returns a Hann window of length n.
func hannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range w {
		w[i] = 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// fftInPlace performs an in-place Cooley-Tukey FFT.
func fftInPlace(x []complex128) {
	n := len(x)
	if n <= 1 {
		return
	}

	// Bit-reversal
	bits := 0
	for tmp := n >> 1; tmp > 0; tmp >>= 1 {
		bits++
	}
	for i := 0; i < n; i++ {
		j := bitReverse(i, bits)
		if j > i {
			x[i], x[j] = x[j], x[i]
		}
	}

	// Butterfly stages
	for s := 2; s <= n; s <<= 1 {
		half := s >> 1
		wBase := complex(math.Cos(-2*math.Pi/float64(s)), math.Sin(-2*math.Pi/float64(s)))
		for k := 0; k < n; k += s {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				t := w * x[k+j+half]
				u := x[k+j]
				x[k+j] = u + t
				x[k+j+half] = u - t
				w *= wBase
			}
		}
	}
}

// bitReverse reverses the bits of v using the given number of bits.
func bitReverse(v, bits int) int {
	result := 0
	for i := 0; i < bits; i++ {
		result = (result << 1) | (v & 1)
		v >>= 1
	}
	return result
}

// max returns the maximum of two int64 values.
func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// NewVAD wraps NewVAD to implement the VAD interface factory.
// It satisfies the vad.New signature.
func init() {
	// The module-level New function is defined in vad.go
}
