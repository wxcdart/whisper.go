# GGUF Format Whisper Models Available

## Discovery

While investigating how whisper.cpp handles models, found that:

1. **whisper.cpp Official Distribution**: Uses their own `ggml` binary format (.bin files)
   - Not GGUF - it's an older custom format
   - Pre-converted from OpenAI PyTorch models
   - Downloaded via `download-ggml-model.sh` script
   - No official GGUF support in whisper.cpp yet

2. **GGUF Conversions Available**: Community has created GGUF versions on HuggingFace
   - Search: "whisper GGUF" on huggingface.co
   - Multiple repos with GGUF-format models
   - Compatible with our Go implementation

## Available GGUF Repositories

Found the following GGUF whisper models on HuggingFace:

```
vonjack/whisper-large-v3-gguf
OllmOne/whisper-medium-GGUF
OllmOne/whisper-large-v3-GGUF
FL33TW00D-HF/whisper-tiny        (likely has GGUF option)
FL33TW00D-HF/whisper-base        (likely has GGUF option)
FL33TW00D-HF/whisper-small       (likely has GGUF option)
Coconuty/Whispertales_model_v4_q8_0.GGUF
```

## How whisper.cpp Works

whisper.cpp uses a 3-step process for models:

1. **Download Official PyTorch Models**
   ```bash
   # Downloads from https://github.com/openai/whisper/
   ```

2. **Convert to ggml Format**
   ```bash
   # Uses convert-pt-to-ggml.py script
   python models/convert-pt-to-ggml.py <pytorch-model> <output>
   ```

3. **Download Pre-Converted Models**
   ```bash
   # From HuggingFace repo ggerganov/whisper.cpp
   ./models/download-ggml-model.sh base.en
   ```

## Our Implementation Path

### Current Status
- ✅ whisper.go built for GGUF format (modern, standardized)
- ❌ .bin format from official whisper.cpp not supported (legacy, complex)
- ✅ GGUF models available from community (compatible!)

### Solution
Use GGUF models from community rather than official .bin models:

1. Find appropriate GGUF repo from the list above
2. Download model: `./download-model <model-name>`
3. Transcribe: `./whisper -m ggml-<model>.gguf -f audio.wav`

### Alternative: Create .bin to GGUF Converter

Since many users have .bin files, we could offer conversion:

```go
// cmd/convert-ggml/main.go
// Wrapper around whisper.cpp's convert-pt-to-ggml.py
// Or: reverse-engineer .bin format and convert to GGUF
```

## Recommendation

**Use GGUF models** (easier, aligned with whisper.go):
- Community GGUF repos available
- No format compatibility issues
- Works with existing code

**Document both options in README**:
- How to find/use GGUF models
- Optional: How to convert .bin files if needed

## Next Steps

1. Pick a small GGUF model from available repos
2. Test download and transcription end-to-end
3. Document working setup in README with examples
4. Consider adding conversion tool as future enhancement
