#include "vocab_json.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
static std::string read_all_utf8(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open file: " + p.string());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void skip_ws(const std::string& s, size_t& i) {
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      ++i;
      continue;
    }
    break;
  }
}

static std::string parse_json_string(const std::string& s, size_t& i) {
  if (i >= s.size() || s[i] != '"') throw std::runtime_error("Expected JSON string");
  ++i;
  std::string out;
  while (i < s.size()) {
    char c = s[i++];
    if (c == '"') break;
    if (c == '\\') {
      if (i >= s.size()) throw std::runtime_error("Invalid escape");
      char e = s[i++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out.push_back(e);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          // For this vocab.json, we don't need unicode escapes; keep a minimal impl.
          // If it ever appears, surface an explicit error.
          throw std::runtime_error("Unsupported \\u escape in vocab.json");
        default:
          throw std::runtime_error("Unsupported escape in vocab.json");
      }
    } else {
      out.push_back(c);
    }
  }
  return out;
}

static int64_t parse_json_int(const std::string& s, size_t& i) {
  skip_ws(s, i);
  bool neg = false;
  if (i < s.size() && s[i] == '-') {
    neg = true;
    ++i;
  }
  if (i >= s.size() || s[i] < '0' || s[i] > '9') throw std::runtime_error("Expected JSON int");
  int64_t v = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
    v = v * 10 + (s[i] - '0');
    ++i;
  }
  return neg ? -v : v;
}
}  // namespace

Vocab load_vocab_json_with_star(const std::filesystem::path& vocab_json_path) {
  const std::string s = read_all_utf8(vocab_json_path);
  size_t i = 0;
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '{') throw std::runtime_error("vocab.json must be a JSON object");
  ++i;

  Vocab v;
  int64_t max_id = -1;
  while (true) {
    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("Unexpected EOF in vocab.json");
    if (s[i] == '}') {
      ++i;
      break;
    }
    const std::string key = parse_json_string(s, i);
    skip_ws(s, i);
    if (i >= s.size() || s[i] != ':') throw std::runtime_error("Expected ':' in vocab.json");
    ++i;
    const int64_t val = parse_json_int(s, i);
    v.token_to_id[key] = val;
    v.id_to_token[val] = key;
    if (val > max_id) max_id = val;

    skip_ws(s, i);
    if (i >= s.size()) throw std::runtime_error("Unexpected EOF in vocab.json");
    if (s[i] == ',') {
      ++i;
      continue;
    }
    if (s[i] == '}') {
      ++i;
      break;
    }
    throw std::runtime_error("Expected ',' or '}' in vocab.json");
  }

  // Append <star> as an extra label (torchaudio style).
  v.star_id = max_id + 1;
  v.token_to_id["<star>"] = v.star_id;
  v.id_to_token[v.star_id] = "<star>";
  return v;
}

