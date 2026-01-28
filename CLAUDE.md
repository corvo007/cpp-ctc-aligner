# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

cpp-ort-aligner is a standalone C++ tool for CTC forced alignment of subtitles to audio. It uses ONNX Runtime for Wav2Vec2-CTC inference and supports multiple audio formats via miniaudio.

## Build Commands

### Windows (recommended)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

### Manual build (Developer Command Prompt)
```bat
cmake -S . -B build -G Ninja -DUSE_SYSTEM_ORT=ON -DONNXRUNTIME_DIR=models\onnxruntime-win-x64-1.23.2
cmake --build build -j 8
```

Output: `build-Release-Ninja-Multi-Config/Release/cpp-ort-aligner.exe`

### CI smoke test
The binary returns exit code 2 when invoked without arguments (usage message).

## Architecture

### Processing Pipeline
```
Audio File → decode_audio_to_16k_mono() → generate_emissions_ort() → forced_align() → merge_repeats_str() → get_spans_str() → postprocess_results() → Output SRT/JSON
```

### Key Components

**Audio Processing** (`audio_decode.cpp`)
- Uses miniaudio (header-only) for decoding MP3/WAV/FLAC
- Resamples to 16kHz mono float samples

**Emissions Generation** (`emissions.cpp`)
- Runs ONNX Runtime inference on Wav2Vec2-CTC model
- Windowed processing (30s windows, 2s context overlap)
- Batched inference support

**Forced Alignment** (`forced_align.cpp`)
- Viterbi-style CTC forced alignment
- Produces frame-level path and per-frame scores

**Span Alignment** (`span_align.cpp`)
- `merge_repeats_str()`: Collapses repeated tokens in path
- `get_spans_str()`: Maps tokens to frame spans

**Text Preprocessing** (`text_preprocess.cpp`)
- Tokenizes text for CTC alignment
- Handles `<star>` token insertion between segments

**Romanization** (CJK support)
- `kana_romaji.cpp`: Japanese kana → romaji
- `kanji_pinyin.cpp`: Chinese kanji → pinyin (requires `Chinese_to_Pinyin.txt`)
- `hangul_romaji.cpp`: Korean hangul → romaji

**I/O**
- `srt_io.cpp`: SRT file parsing/writing
- `json_io.cpp`: JSON input/output format
- `vocab_json.cpp`: Loads model vocabulary

### Data Structures

- `Emissions`: Frame-level log probabilities (T×C matrix)
- `SrtSegment`: Subtitle segment with timing and confidence score
- `WordTimestamp`: Per-word/character timing result
- `SegmentStr`/`SegmentSpanStr`: Intermediate alignment structures

## Dependencies

- **ONNX Runtime 1.23.2**: Bundled in `models/onnxruntime-win-x64-1.23.2/`
- **nlohmann/json**: Header-only in `include/nlohmann/json.hpp`
- **miniaudio**: Header-only in `include/miniaudio.h`

## Model Files

Required in `models/mms-300m-1130-forced-aligner/`:
- `model.onnx` + `model.onnx.data`
- `vocab.json`

For CJK romanization: `Chinese_to_Pinyin.txt` (bundled in release packages, placed next to exe)
