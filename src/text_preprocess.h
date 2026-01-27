#pragma once

#include <string>
#include <vector>

struct PreprocessResult {
  std::vector<std::string> tokens_starred;
  std::vector<std::string> text_starred;
  std::string full_text;
};

// Replicates ctc_forced_aligner.text_utils.preprocess_text as used by align.py.
// - full_text: already concatenated with single spaces between SRT segments
// - language: ISO 639-3 code (e.g. "jpn")
// - romanize: if true, use C++ romanization for CJK languages
PreprocessResult preprocess_text_cpp(const std::string& full_text, const std::string& language, bool romanize);
