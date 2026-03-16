# Model Download & Integration Test Report

**Date**: March 16, 2026  
**Status**: ⚠️ Format mismatch detected

## Summary

Successfully downloaded the tiny model (74.1 MB) using the new `download-model` utility, but encountered a file format issue during transcription testing.

## What Worked

✅ **download-model CLI**
- Built successfully with `make download-model`
- Listed all 30+ available models with `-l` flag
- Downloaded 74.1 MB tiny model from HuggingFace in ~60s
- Saved to `models/ggml-tiny.bin` as expected

✅ **Test Audio Generation**
- Created synthetic `testdata/test.wav` (2s @ 16kHz, mono, 16-bit)
- Used for integration testing

✅ **whisper CLI Build**
- Built successfully with `make whisper`
- Ready for transcription

## Issue Discovered

❌ **Format Mismatch**

The models available from `ggerganov/whisper.cpp` on HuggingFace are in the **old ggml binary format** (`.bin`), not the newer **GGUF format** that our implementation expects.

**File Header Analysis:**
```
File: models/ggml-tiny.bin (74.1 MB)
Magic: "lmgg" (old ggml format)
Our parser expects: "GGUF"
Result: Load failed - invalid magic error
```

**Error Output:**
```
Loading audio from testdata/test.wav
Loading model from models/ggml-tiny.bin
error: failed to load model: load GGUF: gguf: invalid magic
```

## Root Cause

The whisper.cpp project has models in two formats:
- **Old format (.bin)**: Uses "lmgg" magic, older ggml format
- **New format (.gguf)**: Uses "GGUF" magic, GGUF v3 format

Our implementation built support for the newer GGUF format, but the official whisper.cpp HuggingFace repo primarily distributes old format models. (They've since transitioned but kept the old ones available.)

## Solutions (Pick One)

### Option 1: Add Old ggml Format Support (Recommended)
- Implement old ggml format parser in `internal/gguf/old_format.go`
- Add format detection in `Open()` function
- Pros: Works with existing models, backward compatible
- Cons: ~500 LOC additional code

### Option 2: Find GGUF Models
- Search for pre-converted GGUF versions on HuggingFace
- Possible repos: Different maintainers, community conversions
- Pros: Uses our current code
- Cons: Harder to find, may not have all model sizes

### Option 3: Create Conversion Tool
- Add converter from old .bin → GGUF in quantize tool
- Requires understanding old format structure
- Pros: Maximum flexibility
- Cons: Complex, requires upstream format knowledge

### Option 4: Use Convert Script
- Use official whisper.cpp Python conversion tools
- Pros: Proven to work
- Cons: Requires Python, adds build dependency

## Next Steps

Recommend **Option 1** (old format support) as it:
- Enables use of official models immediately
- Aligns with whisper.cpp ecosystem
- Minimal code addition
- No external dependencies

## Test Results Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Build system | ✅ Pass | make build/test/clean all work |
| download-model | ✅ Pass | Successfully downloads models |
| Test audio | ✅ Pass | 2s synthetic WAV created |
| Model loading | ❌ Fail | Format mismatch (ggml vs GGUF) |
| Transcription | ⏸️ Blocked | Depends on model loading fix |

## Files Modified

- `cmd/download-model/main.go` — Fixed to download `.bin` files (correct format available)
- `testdata/test.wav` — Created synthetic 2-second test audio

## Commands Used

```bash
# Build and list models
make download-model
./download-model -l

# Download tiny model
./download-model -o models tiny
# Result: models/ggml-tiny.bin (74.1 MB)

# Generate test audio
python3 << 'EOF'
import wave, math
sample_rate = 16000
duration = 2
frequency = 440
samples = [int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate)) 
           for i in range(sample_rate * duration)]
with wave.open('testdata/test.wav', 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    for s in samples:
        f.writeframes(s.to_bytes(2, 'little', signed=True))
EOF

# Test transcription (fails with format error)
./whisper -m models/ggml-tiny.bin -f testdata/test.wav
```

## Conclusion

The `download-model` utility is **working correctly**. The issue is a format mismatch between our GGUF-only parser and the old ggml format models in the official distribution.

**Action Required**: Implement old format parser to enable end-to-end testing with real models.

---

**End of Test Report**
