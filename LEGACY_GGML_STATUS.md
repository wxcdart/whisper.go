# Legacy GGML Format Support Status

## Current Status: PARTIAL ✅ LOADING, ❌ INFERENCE

The legacy ggml.bin format (magic `0x67676d6c` / "ggml") is now **successfully parsed and loaded**, but **inference produces empty output** due to apparent model weight or architecture incompatibility.

## What Works ✅

- **Binary Format Parsing**: Strict parser correctly reads magic, header, metadata blocks, and tensor records
- **Tensor Discovery**: All 167 tensors from `ggml-tiny.bin` discovered and indexed correctly
- **Tensor Name Resolution**: Legacy tensor names (`attn.*`, `attn_ln.*`) correctly mapped to decoder expectations
- **Tensor Loading**: Weights successfully loaded into memory with reasonable value ranges
- **Encoder Execution**: Encoder produces output with expected shape [1500, 384] and reasonable value ranges (-17.49 to 17.39)
- **Model Inference**: Decoder forward passes run without errors
- **No Regressions**: All existing unit tests pass

## What Doesn't Work ❌

- **Token Generation**: Decoder always generates EOT (end-of-transcript) token as first predicted token after prompt
- **Inference Correctness**: Results in empty transcript output (no text generated)
- **Consistent Issue**: Occurs regardless of audio input, suggesting systematic model issue rather than input-specific problem

## Evidence

### Model Load Success
```
[DEBUG] Tensor resolution: "decoder.blocks.0.self_attn_ln.weight" -> "decoder.blocks.0.attn_ln.weight" ✓
[DEBUG] Token embedding shape=[51865 384] [sample min=-0.0753, max=0.1035] ✓
[DEBUG] Parsed 167 tensors from ggml-tiny.bin ✓
```

### Inference Failure
```
[DEBUG] decodeGreedy attempt 0: prompt tokens=[50258, 50259, 50359]
[DEBUG] Encoder output: shape=[1500 384] [min=-17.4866, max=17.3866, zeros=0/576000] ✓
[DEBUG] First forward: logits len=51865, max_logit=6.4210 at idx=50257 (EOT token)
[DEBUG] Top 5 logits: [50257:6.42] [1230:4.90] [1919:4.82] [522:4.64] [19347:4.45]
→ Sampled first token: EOT (50257)
→ Output: empty transcript
```

## Analysis

The consistent generation of EOT as the first token suggests:

1. **Model weights are significantly different** from fine-tuned GGUF versions
   - Embeddings load correctly but may have scale/normalization mismatch
   - Attention and linear layer weights may be in different format
   - Cross-attention layer may not be properly using encoder output

2. **Legacy ggml format conversions may have issues**
   - The original `ggml-tiny.bin` model may be corrupted, incomplete, or from different training run
   - Model may have been converted with incorrect parameter settings

3. **Less likely: Inference logic issues**
   - Encoder output shape is correct
   - Decoder runs without errors
   - but systematic preference for EOT suggests data issue rather than code issue

## Known Issues

- No access to reference implementation output for comparison
- Cannot determine if issue is model-specific or format-wide without testing other ggml models
- Legacy ggml format lacks official specification; parser based on reverse-engineering

## Recommendations

### Short Term (Current)
1. **Use GGUF models** - These work correctly and are the modern standard
2. **Document as known limitation** - Note that legacy ggml support is parsing-only, not inference-ready

### Medium Term
1. **Test with different ggml models** - Determine if issue affects all ggml binaries
2. **Compare with whisper.cpp** - Test same binary with reference implementation
3. **Add validation tests** - Ensure inference produces non-EOT tokens first

### Long Term
1. **Access official spec** - Obtain format documentation from ggerganov/whisper.cpp
2. **Implement conversion** - Create ggml→GGUF converter using reference tools
3. **Full compatibility** - Once root cause identified, fix model loading/inference

## Test Results

- ✅ All 10 internal packages pass unit tests
- ✅ `go build ./...` succeeds
- ✅ Model loads without errors
- ✅ Transcription completes in ~7 seconds
- ❌ Output is empty (no text generated)

## File Changes

Modified files for legacy format support:
- `internal/ggml/ggml.go` - Strict legacy parser with tensor discovery
- `internal/model/compat.go` - Tensor name mapping for legacy conventions
- `internal/model/encoder.go` - Shape normalization for legacy conv bias tensors

## References

- Original whisper.cpp: https://github.com/ggerganov/whisper.cpp
- GGML format: https://github.com/ggerganov/ggml/blob/master/include/ggml.h
- Whisper models: https://huggingface.co/ggerganov/whisper.cpp

---

**Status**: Ready for investigation/debugging or component replacement with proven GGUF models.
**Priority**: Low (GGUF models work well as alternative)
