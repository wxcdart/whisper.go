// Command quantize converts a GGUF model to a quantised format.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/whispergo/whisper.go/internal/gguf"
)

func main() {
	var (
		model   string
		output  string
		typ     string
		verbose bool
	)

	flag.StringVar(&model, "m", "", "input GGUF file (required)")
	flag.StringVar(&model, "model", "", "input GGUF file (required)")
	flag.StringVar(&output, "o", "", "output GGUF file (required)")
	flag.StringVar(&output, "output", "", "output GGUF file (required)")
	flag.StringVar(&typ, "t", "", "quantisation type: q4_0, q4_1, q5_0, q5_1, q8_0, f16 (required)")
	flag.StringVar(&typ, "type", "", "quantisation type: q4_0, q4_1, q5_0, q5_1, q8_0, f16 (required)")
	flag.BoolVar(&verbose, "v", false, "print per-tensor info")
	flag.BoolVar(&verbose, "verbose", false, "print per-tensor info")
	flag.Usage = usage
	flag.Parse()

	if model == "" || output == "" || typ == "" {
		flag.Usage()
		os.Exit(1)
	}

	targetType, err := parseQuantType(typ)
	if err != nil {
		log.Fatalf("quantize: %v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	if verbose {
		if err := quantizeVerbose(ctx, model, output, targetType); err != nil {
			log.Fatalf("quantize: %v", err)
		}
		return
	}

	if err := gguf.QuantizeFile(ctx, model, output, targetType); err != nil {
		log.Fatalf("quantize: %v", err)
	}
}

// quantizeVerbose opens src, quantises each tensor individually and prints
// per-tensor progress before writing dst.
func quantizeVerbose(ctx context.Context, src, dst string, targetType gguf.QuantType) error {
	f, err := gguf.Open(ctx, src)
	if err != nil {
		return fmt.Errorf("open %s: %w", src, err)
	}
	defer f.Close() //nolint:errcheck

	out := gguf.NewWritableFile()
	names := f.TensorNames()

	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return err
		}

		data, shape, err := f.Tensor(ctx, name)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}

		srcDtype := gguf.QuantF32
		if st, ok := f.TensorType(name); ok {
			srcDtype = st
		}
		srcBytes := len(data) * 4

		var (
			dstDtype gguf.QuantType
			qdata    []byte
		)

		shape64 := make([]uint64, len(shape))
		for i, s := range shape {
			shape64[i] = uint64(s)
		}

		if cliShouldQuantize(name) {
			dstDtype = targetType
			qdata, err = cliApplyQuant(data, targetType)
		} else {
			dstDtype = gguf.QuantF32
			qdata = cliFloat32sToBytes(data)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}

		fmt.Printf("quantizing %s: %s → %s (%d → %d bytes)\n",
			name, srcDtype, dstDtype, srcBytes, len(qdata))

		out.AddTensor(name, shape64, dstDtype, qdata)
	}

	return gguf.WriteFile(ctx, dst, out)
}

// cliShouldQuantize mirrors the internal shouldQuantize rule.
func cliShouldQuantize(name string) bool {
	lower := strings.ToLower(name)
	if strings.Contains(lower, "norm") || strings.Contains(lower, ".ln") {
		return false
	}
	if strings.HasSuffix(lower, ".bias") {
		return false
	}
	return strings.HasSuffix(lower, ".weight")
}

// cliApplyQuant dispatches to the exported quantisation kernels.
func cliApplyQuant(data []float32, t gguf.QuantType) ([]byte, error) {
	switch t {
	case gguf.QuantF32:
		return cliFloat32sToBytes(data), nil
	case gguf.QuantF16:
		return gguf.QuantizeF16(data), nil
	case gguf.QuantQ4_0:
		return gguf.QuantizeQ4_0(data), nil
	case gguf.QuantQ4_1:
		return gguf.QuantizeQ4_1(data), nil
	case gguf.QuantQ5_0:
		return gguf.QuantizeQ5_0(data), nil
	case gguf.QuantQ5_1:
		return gguf.QuantizeQ5_1(data), nil
	case gguf.QuantQ8_0:
		return gguf.QuantizeQ8_0(data), nil
	default:
		return nil, fmt.Errorf("unsupported quant type %d", t)
	}
}

// cliFloat32sToBytes serialises a float32 slice as little-endian IEEE-754.
func cliFloat32sToBytes(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func parseQuantType(s string) (gguf.QuantType, error) {
	switch s {
	case "f32":
		return gguf.QuantF32, nil
	case "f16":
		return gguf.QuantF16, nil
	case "q4_0":
		return gguf.QuantQ4_0, nil
	case "q4_1":
		return gguf.QuantQ4_1, nil
	case "q5_0":
		return gguf.QuantQ5_0, nil
	case "q5_1":
		return gguf.QuantQ5_1, nil
	case "q8_0":
		return gguf.QuantQ8_0, nil
	default:
		return 0, fmt.Errorf("unknown quantisation type %q (choose: q4_0, q4_1, q5_0, q5_1, q8_0, f16)", s)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, "Usage: quantize -m <input.gguf> -o <output.gguf> -t <type>\n\n")
	fmt.Fprintf(os.Stderr, "Flags:\n")
	fmt.Fprintf(os.Stderr, "  -m, --model   string   input GGUF file (required)\n")
	fmt.Fprintf(os.Stderr, "  -o, --output  string   output GGUF file (required)\n")
	fmt.Fprintf(os.Stderr, "  -t, --type    string   quantisation type: q4_0, q4_1, q5_0, q5_1, q8_0, f16 (required)\n")
	fmt.Fprintf(os.Stderr, "  -v, --verbose          print per-tensor info\n")
}
