#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

struct Vocab {
  std::unordered_map<std::string, int64_t> token_to_id;
  std::unordered_map<int64_t, std::string> id_to_token;
  int64_t star_id = -1;
};

Vocab load_vocab_json_with_star(const std::filesystem::path& vocab_json_path);

