cmake_minimum_required(VERSION 3.20)
project(cpp_ort_aligner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_SYSTEM_ORT "Use system-installed ONNX Runtime" ON)

# Include directory for nlohmann/json and other headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(cpp-ort-aligner
  src/cli_args.cpp
  src/main.cpp
  src/audio_decode.cpp
  src/emissions.cpp
  src/forced_align.cpp
  src/hangul_romaji.cpp
  src/json_io.cpp
  src/kana_romaji.cpp
  src/kanji_pinyin.cpp
  src/span_align.cpp
  src/postprocess.cpp
  src/srt_io.cpp
  src/text_preprocess.cpp
  src/vocab_json.cpp
)

if (USE_SYSTEM_ORT)
  # Expect user sets ONNXRUNTIME_DIR to the extracted onnxruntime package root.
  # If not set, default to the bundled ORT under cpp-ort-aligner/models/.
  if (NOT DEFINED ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/models/onnxruntime-win-x64-1.23.2")
  endif()

  # Use absolute paths to avoid generator-dependent relative path quirks on Windows.
  get_filename_component(ONNXRUNTIME_DIR_ABS "${ONNXRUNTIME_DIR}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
  target_include_directories(cpp-ort-aligner PRIVATE "${ONNXRUNTIME_DIR_ABS}/include")
  target_link_directories(cpp-ort-aligner PRIVATE "${ONNXRUNTIME_DIR_ABS}/lib")
  target_link_libraries(cpp-ort-aligner PRIVATE onnxruntime)
endif()

if (MSVC)
  target_compile_options(cpp-ort-aligner PRIVATE /W4 /permissive- /utf-8)
  target_compile_definitions(cpp-ort-aligner PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
  target_compile_options(cpp-ort-aligner PRIVATE -Wall -Wextra -Wpedantic)
endif()
