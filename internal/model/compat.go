package model

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/whispergo/whisper.go/internal/gguf"
	"github.com/whispergo/whisper.go/internal/ml"
)

const whisperHeadDim = 64

func resolveTensorName(f *gguf.File, name string) (string, bool) {
	if _, ok := f.TensorType(name); ok {
		return name, true
	}

	alt := mapTensorName(name)
	if alt == "" {
		return "", false
	}
	if _, ok := f.TensorType(alt); ok {
		return alt, true
	}

	return "", false
}

func mapTensorName(name string) string {
	simple := map[string]string{
		"encoder.conv1.weight":          "model.encoder.conv1.weight",
		"encoder.conv1.bias":            "model.encoder.conv1.bias",
		"encoder.conv2.weight":          "model.encoder.conv2.weight",
		"encoder.conv2.bias":            "model.encoder.conv2.bias",
		"encoder.positional_embedding":  "model.encoder.embed_positions.weight",
		"encoder.ln_post.weight":        "model.encoder.layer_norm.weight",
		"encoder.ln_post.bias":          "model.encoder.layer_norm.bias",
		"decoder.token_embedding.weight": "model.decoder.embed_tokens.weight",
		"decoder.positional_embedding":  "model.decoder.embed_positions.weight",
		"decoder.ln.weight":             "model.decoder.layer_norm.weight",
		"decoder.ln.bias":               "model.decoder.layer_norm.bias",
	}
	if mapped, ok := simple[name]; ok {
		return mapped
	}

	if strings.HasPrefix(name, "encoder.blocks.") {
		i, suffix, ok := parseIndexedSuffix(name, "encoder.blocks.")
		if !ok {
			return ""
		}
		base := "model.encoder.layers." + strconv.Itoa(i) + "."
		switch suffix {
		case "attn_ln.weight":
			return base + "self_attn_layer_norm.weight"
		case "attn_ln.bias":
			return base + "self_attn_layer_norm.bias"
		case "attn.query.weight":
			return base + "self_attn.q_proj.weight"
		case "attn.query.bias":
			return base + "self_attn.q_proj.bias"
		case "attn.key.weight":
			return base + "self_attn.k_proj.weight"
		case "attn.value.weight":
			return base + "self_attn.v_proj.weight"
		case "attn.value.bias":
			return base + "self_attn.v_proj.bias"
		case "attn.out.weight":
			return base + "self_attn.out_proj.weight"
		case "attn.out.bias":
			return base + "self_attn.out_proj.bias"
		case "mlp_ln.weight":
			return base + "final_layer_norm.weight"
		case "mlp_ln.bias":
			return base + "final_layer_norm.bias"
		case "mlp.0.weight":
			return base + "fc1.weight"
		case "mlp.0.bias":
			return base + "fc1.bias"
		case "mlp.2.weight":
			return base + "fc2.weight"
		case "mlp.2.bias":
			return base + "fc2.bias"
		}
	}

	if strings.HasPrefix(name, "decoder.blocks.") {
		i, suffix, ok := parseIndexedSuffix(name, "decoder.blocks.")
		if !ok {
			return ""
		}
		base := "model.decoder.layers." + strconv.Itoa(i) + "."
		switch suffix {
		case "self_attn_ln.weight":
			return base + "self_attn_layer_norm.weight"
		case "self_attn_ln.bias":
			return base + "self_attn_layer_norm.bias"
		case "self_attn.query.weight":
			return base + "self_attn.q_proj.weight"
		case "self_attn.query.bias":
			return base + "self_attn.q_proj.bias"
		case "self_attn.key.weight":
			return base + "self_attn.k_proj.weight"
		case "self_attn.value.weight":
			return base + "self_attn.v_proj.weight"
		case "self_attn.value.bias":
			return base + "self_attn.v_proj.bias"
		case "self_attn.out.weight":
			return base + "self_attn.out_proj.weight"
		case "self_attn.out.bias":
			return base + "self_attn.out_proj.bias"
		case "cross_attn_ln.weight":
			return base + "encoder_attn_layer_norm.weight"
		case "cross_attn_ln.bias":
			return base + "encoder_attn_layer_norm.bias"
		case "cross_attn.query.weight":
			return base + "encoder_attn.q_proj.weight"
		case "cross_attn.query.bias":
			return base + "encoder_attn.q_proj.bias"
		case "cross_attn.key.weight":
			return base + "encoder_attn.k_proj.weight"
		case "cross_attn.value.weight":
			return base + "encoder_attn.v_proj.weight"
		case "cross_attn.value.bias":
			return base + "encoder_attn.v_proj.bias"
		case "cross_attn.out.weight":
			return base + "encoder_attn.out_proj.weight"
		case "cross_attn.out.bias":
			return base + "encoder_attn.out_proj.bias"
		case "mlp_ln.weight":
			return base + "final_layer_norm.weight"
		case "mlp_ln.bias":
			return base + "final_layer_norm.bias"
		case "mlp.0.weight":
			return base + "fc1.weight"
		case "mlp.0.bias":
			return base + "fc1.bias"
		case "mlp.2.weight":
			return base + "fc2.weight"
		case "mlp.2.bias":
			return base + "fc2.bias"
		}
	}

	return ""
}

func parseIndexedSuffix(name, prefix string) (int, string, bool) {
	rest := strings.TrimPrefix(name, prefix)
	parts := strings.SplitN(rest, ".", 2)
	if len(parts) != 2 {
		return 0, "", false
	}
	i, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, "", false
	}
	return i, parts[1], true
}

func getMetaAny(f *gguf.File, keys ...string) (uint32, bool) {
	for _, k := range keys {
		if v, ok := f.MetaUint32(k); ok {
			return v, true
		}
	}
	return 0, false
}

func inferLayerCount(f *gguf.File, sampleNameFmt string) (int, bool) {
	count := 0
	for {
		name := fmt.Sprintf(sampleNameFmt, count)
		if _, ok := resolveTensorName(f, name); !ok {
			break
		}
		count++
	}
	if count == 0 {
		return 0, false
	}
	return count, true
}

func loadTensorShape(ctx context.Context, f *gguf.File, name string) ([]int, error) {
	resolved, ok := resolveTensorName(f, name)
	if !ok {
		return nil, fmt.Errorf("model: load tensor %q: not found", name)
	}
	_, shape, err := f.Tensor(ctx, resolved)
	if err != nil {
		return nil, fmt.Errorf("model: load tensor %q (resolved %q): %w", name, resolved, err)
	}
	return shape, nil
}

func loadTensorAuto(ctx context.Context, f *gguf.File, name string) (ml.Tensor, string, error) {
	resolved, ok := resolveTensorName(f, name)
	if !ok {
		return ml.Tensor{}, "", fmt.Errorf("model: load tensor %q: not found", name)
	}
	data, shape, err := f.Tensor(ctx, resolved)
	if err != nil {
		return ml.Tensor{}, resolved, fmt.Errorf("model: load tensor %q (resolved %q): %w", name, resolved, err)
	}
	return ml.From(data, shape...), resolved, nil
}
