# Old GGML Format Implementation Status

## Overview

Attempted to implement support for old ggml binary format (magic: "lmgg") to enable use of models from the official ggerganov/whisper.cpp HuggingFace repository.

## Status: INCOMPLETE

The old ggml format reverse-engineering proved more complex than initially anticipated. The binary format structure is more sophisticated than a simple header + tensors layout.

## What Was Done

1. **Format Detection** — Added magic number detection in `Open()` function to identify old ggml files ("lmgg") vs new GGUF files ("GGUF")
2. **Partial Parser** — Implemented `oldformat.go` with initial structure parsing attempt
3. **Type Mapping** — Created mapping from old ggml type codes to GGUF tensor types

## Why It's Incomplete

Initial analysis of the binary structure revealed:
- Magic: "lmgg" ✓
- Version field: Readable ✓  
- Tensor count: Parsing becomes misaligned ✗
- Subsequent parsing: Fails with EOF errors

The tensor metadata layout in the old format appears to differ from what was documented/inferred.

## Solutions

### Option 1: Use GGUF Models (Recommended, Immediate)
Most practical approach for current implementation:
```bash
# Find GGUF-converted versions on HuggingFace
# Search for repos with ggml models converted to GGUF format
# Example: openai/whisper-tiny (if GGUF versions available)
```

### Option 2: Complete Old Format Implementation (Complex)
Would require:
- Access to official format specification from whisper.cpp source
- Detailed binary format documentation
- Extensive testing and validation
- ~500-1000 LOC additional implementation

### Option 3: Conversion Tool
Create a .bin → .gguf converter:
- Leverage existing whisper.cpp utilities
- Could be Python wrapper or Go implementation
- Allows using existing model distribution

### Option 4: Python Integration
Implement Python subprocess calls:
- Use official `convert-to-gguf.py` from whisper.cpp
- Run as pre-processing step
- Ensures format correctness

## Current Code

**File**: `internal/gguf/oldformat.go`
- Format detection: ✓ Working
- Type mapping: ✓ Implemented
- Tensor parsing: ✗ Incomplete
- Index building: ✓ Implemented

**File**: `internal/gguf/gguf.go`  
- Auto-detection switch: ✓ Working
- GGUF parsing: ✓ Unchanged/working
- Old format routing: ✓ Working
- Old format handling: ✗ Needs format fix

## Recommendation

**Do not spend additional time on reverse-engineering the old format.** Instead:

1. **Immediate**: Use GGUF-format models when available
2. **Short-term**: Create simple conversion wrapper script
3. **Long-term**: Consider integrating official whisper.cpp conversion tools

The current auto-detection infrastructure is in place and will work seamlessly once proper old format parsing is available.

## Test Results

```
✓ Format detection compiles
✓ GGUF parsing still works
✗ Old format parsing: EOF errors on tensor 6+
```

---

**Next Steps**: Find or create GGUF-format models, or implement conversion tool.

