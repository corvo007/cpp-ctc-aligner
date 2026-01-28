#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
#include "cli_args.h"
#include "emissions.h"
#include "forced_align.h"
#include "json_io.h"
#include "logger.h"
#include "postprocess.h"
#include "span_align.h"
#include "srt_io.h"
#include "text_preprocess.h"
#include "vocab_json.h"
#include "audio_decode.h"
#include "kanji_pinyin.h"
#include "stacktrace.h"

// Emissions generation now lives in emissions.cpp; keep main minimal.

// Forward declaration
static int run_alignment(int argc, char** argv);

static std::string json_quote_str(const std::string& s) {
  // Minimal JSON string escaping (enough for our debug artifacts).
  std::string out;
  out.reserve(s.size() + 8);
  out.push_back('"');
  for (char c : s) {
    switch (c) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  out.push_back('"');
  return out;
}

int main(int argc, char** argv) {
  try {
    return run_alignment(argc, argv);
  } catch (const Ort::Exception& e) {
    std::cerr << "\n[ERROR] ONNX Runtime error: " << e.what() << "\n";
    std::cerr << stacktrace::capture_string(0) << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "\n[ERROR] " << e.what() << "\n";
    std::cerr << stacktrace::capture_string(0) << "\n";
    return 1;
  } catch (...) {
    std::cerr << "\n[ERROR] Unknown exception occurred\n";
    std::cerr << stacktrace::capture_string(0) << "\n";
    return 1;
  }
}

static int run_alignment(int argc, char** argv) {
  CliArgs args;
  int exit_code = 0;
  if (!parse_cli_args(argc, argv, args, exit_code)) {
    return exit_code;
  }

  Logger log;
  log.set_debug(args.debug);
  if (args.debug && !args.debug_dir.empty()) {
    std::error_code ec;
    fs::create_directories(args.debug_dir, ec);
    log.enable_file(args.debug_dir / "alignment.log");
  }

  const fs::path wav_path = args.audio;
  const std::string language = args.language;
  const bool romanize = args.romanize;
  const int batch_size = args.batch_size;

  // Load kanji pinyin table if using romanization
  if (romanize) {
    const fs::path& pinyin_table_path = args.pinyin_table;
    log.info(std::string("Loading kanji pinyin table from: ") + pinyin_table_path.string());
    if (!kanji::load_pinyin_table(pinyin_table_path.string())) {
      throw std::runtime_error("Failed to load kanji pinyin table from: " + pinyin_table_path.string());
    }
    log.info("Kanji pinyin table loaded successfully");
  }

  const auto audio_samples = decode_audio_to_16k_mono(wav_path);
  {
    std::ostringstream ss;
    ss << "Loaded audio: " << audio_samples.size() << " samples (" << (audio_samples.size() / 16000.0) << " seconds)";
    log.info(ss.str());
  }
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cpp-ort-aligner");
  Ort::SessionOptions opts;
  int num_threads = args.threads;
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads <= 0) num_threads = 4;
    num_threads = std::max(4, (num_threads + 1) / 2);
  }
  opts.SetIntraOpNumThreads(num_threads);
  opts.SetInterOpNumThreads(1);  // Keep inter-op at 1 for better cache locality
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  {
    std::ostringstream ss;
    ss << "ORT threads: " << num_threads << ", graph optimization: ALL";
    log.info(ss.str());
  }

  const fs::path model_onnx = args.model_dir / "model.onnx";
  const fs::path vocab_json_path = args.model_dir / "vocab.json";
#ifdef _WIN32
  Ort::Session session(env, model_onnx.wstring().c_str(), opts);
#else
  Ort::Session session(env, model_onnx.string().c_str(), opts);
#endif

  auto emissions = generate_emissions_ort(
      session,
      audio_samples,
      /*window_seconds=*/30,
      /*context_seconds=*/2,
      /*batch_size=*/batch_size,
      /*star_logp=*/0.0f);

  {
    std::ostringstream ss;
    ss << "emissions shape: [" << emissions.frames << "," << emissions.classes << "] stride_ms=" << emissions.stride_ms;
    log.info(ss.str());
  }

  std::vector<SrtSegment> srt_segments;

  // Read segments from JSON or SRT input
  if (!args.json_input.empty()) {
    if (args.json_input.string() == "-") {
      // Read from stdin
      std::ostringstream ss;
      ss << std::cin.rdbuf();
      srt_segments = parse_json_input(ss.str());
    } else {
      srt_segments = read_json_input(args.json_input);
    }
    {
      std::ostringstream ss;
      ss << "Read " << srt_segments.size() << " segments from JSON input";
      log.info(ss.str());
    }
  } else {
    srt_segments = read_srt_utf8(args.srt);
  }
  const std::vector<SrtSegment> original_segments_for_debug = args.debug ? srt_segments : std::vector<SrtSegment>{};

  std::string full_text;
  for (size_t i = 0; i < srt_segments.size(); ++i) {
    std::string seg_text = srt_segments[i].text;
    for (char& ch : seg_text) {
      if (ch == '\n') ch = ' ';
    }
    while (!seg_text.empty() && std::isspace(static_cast<unsigned char>(seg_text.front()))) seg_text.erase(seg_text.begin());
    while (!seg_text.empty() && std::isspace(static_cast<unsigned char>(seg_text.back()))) seg_text.pop_back();
    if (i) full_text.push_back(' ');
    full_text += seg_text;
  }

  std::vector<std::string> tokens_starred;
  std::vector<std::string> text_starred;
  std::string full_text_for_debug = full_text;

  const auto prep = preprocess_text_cpp(full_text, language, romanize);
  full_text_for_debug = prep.full_text;
  tokens_starred = prep.tokens_starred;
  text_starred = prep.text_starred;

  const auto vocab = load_vocab_json_with_star(vocab_json_path);
  const int64_t star_id = vocab.star_id;
  if (star_id != emissions.classes - 1) {
    throw std::runtime_error(
        "vocab size mismatch: emissions classes=" + std::to_string(emissions.classes) + ", vocab+star=" +
        std::to_string(star_id + 1) + " (check matching model.onnx + vocab.json)");
  }

  std::vector<int64_t> targets;
  targets.reserve(2000);
  // Python: token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
  std::string joined;
  for (const auto& t : tokens_starred) {
    if (!joined.empty()) joined.push_back(' ');
    joined += t;
  }
  size_t pos = 0;
  while (pos < joined.size()) {
    size_t next = joined.find(' ', pos);
    if (next == std::string::npos) next = joined.size();
    const auto piece = joined.substr(pos, next - pos);
    if (!piece.empty()) {
      if (piece == "<star>") {
        targets.push_back(star_id);
      } else {
        auto it = vocab.token_to_id.find(piece);
        if (it != vocab.token_to_id.end()) targets.push_back(it->second);
      }
    }
    pos = next + 1;
  }

  std::vector<int64_t> path;
  std::vector<float> scores;
  forced_align(
      emissions.log_probs.data(),
      emissions.frames,
      emissions.classes,
      targets.data(),
      int64_t(targets.size()),
      /*blank=*/0,
      path,
      scores);

  // Build idx_to_token for spans from vocab.json.
  std::unordered_map<int64_t, std::string> idx_to_token;
  for (const auto& kv : vocab.token_to_id) {
    idx_to_token[kv.second] = kv.first;
  }
  idx_to_token[star_id] = "<star>";

  const auto segments = merge_repeats_str(path, idx_to_token);

  try {
    const auto spans = get_spans_str(tokens_starred, segments, "<blank>");

    auto word_ts = postprocess_results(text_starred, spans, emissions.stride_ms, scores);

      // Map word timestamps back to SRT segments (match Python align.py:map_timestamps_to_srt).
        size_t char_idx = 0;
        for (auto& seg : srt_segments) {
          // Python uses:
          //   seg_text = seg.text.replace("\n", " ").strip()
          //   num_chars = len(seg_text)
          //   space_offset = 1 if char_idx > 0 else 0
          //   if space_offset>0 and word_timestamps[char_idx]["text"].strip()=="" then char_idx += 1
          //   start_time = word_timestamps[char_idx]["start"]
          //   end_idx = min(char_idx + num_chars - 1, len(word_timestamps)-1)
          //   end_time = word_timestamps[end_idx]["end"]
          //
          // IMPORTANT: `num_chars` is counted in *Python characters* (Unicode codepoints),
          // not bytes. Our `word_ts` tokens are UTF-8 strings, so we must count UTF-8 codepoints
          // in `seg_text`, not `seg_text.size()` bytes.
          auto utf8_codepoints = [](const std::string& s) -> size_t {
            size_t count = 0;
            for (size_t i = 0; i < s.size();) {
              unsigned char c = static_cast<unsigned char>(s[i]);
              if (c < 0x80) {
                i += 1;
              } else if ((c >> 5) == 0x6) {
                i += 2;
              } else if ((c >> 4) == 0xE) {
                i += 3;
              } else if ((c >> 3) == 0x1E) {
                i += 4;
              } else {
                // invalid byte, advance to avoid infinite loop
                i += 1;
              }
              count += 1;
            }
            return count;
          };

          std::string seg_text = seg.text;
          for (char& ch : seg_text) {
            if (ch == '\n') ch = ' ';
          }
          // strip ASCII whitespace like Python's .strip()
          while (!seg_text.empty() && std::isspace(static_cast<unsigned char>(seg_text.front()))) {
            seg_text.erase(seg_text.begin());
          }
          while (!seg_text.empty() && std::isspace(static_cast<unsigned char>(seg_text.back()))) {
            seg_text.pop_back();
          }

          const size_t num_chars = utf8_codepoints(seg_text);
          if (num_chars == 0 || char_idx >= word_ts.size()) {
            continue;
          }

          if (char_idx > 0 && char_idx < word_ts.size()) {
            // Python: if word_timestamps[char_idx]["text"].strip() == "": char_idx += 1
            std::string t = word_ts[char_idx].text;
            // trim ASCII whitespace for strip()
            size_t l = 0;
            while (l < t.size() && std::isspace(static_cast<unsigned char>(t[l]))) l++;
            size_t r = t.size();
            while (r > l && std::isspace(static_cast<unsigned char>(t[r - 1]))) r--;
            if (r <= l) {
              char_idx += 1;
            }
          }
          if (char_idx >= word_ts.size()) {
            continue;
          }

          const size_t start_idx = char_idx;
          const auto& first = word_ts[char_idx];
          const size_t end_idx = std::min(char_idx + num_chars - 1, word_ts.size() - 1);
          const auto& last = word_ts[end_idx];
          seg.start_sec = first.start_sec;
          seg.end_sec = last.end_sec;

          // Compute average confidence score from word scores in this segment
          // Match Python: convert log_prob to linear probability, then average
          std::vector<float> token_probs;
          token_probs.reserve(end_idx - start_idx + 1);
          for (size_t wi = start_idx; wi <= end_idx && wi < word_ts.size(); ++wi) {
            const auto& wt = word_ts[wi];
            // wt.score is sum of log_prob over frames for this token
            float token_total_log_prob = wt.score;
            // Approximate frame count (stride is typically 20ms)
            float token_duration = static_cast<float>(wt.end_sec - wt.start_sec);
            int n_frames = std::max(1, static_cast<int>(token_duration / 0.02f));
            // Average log_prob per frame
            float token_avg_log_prob = token_total_log_prob / static_cast<float>(n_frames);
            // Convert to linear probability (0.0 - 1.0)
            float token_prob = std::min(1.0f, std::exp(token_avg_log_prob));
            token_probs.push_back(token_prob);
          }
          // Final score is arithmetic mean of token probabilities
          if (!token_probs.empty()) {
            float prob_sum = 0.0f;
            for (float p : token_probs) prob_sum += p;
            seg.score = prob_sum / static_cast<float>(token_probs.size());
          } else {
            seg.score = 0.0f;
          }

          char_idx = end_idx + 1;
        }

        // Write output in JSON or SRT format
        if (!args.json_output.empty()) {
          if (args.json_output.string() == "-") {
            // Write to stdout
            std::cout << format_json_output(srt_segments, 0.0);
          } else {
            write_json_output(args.json_output, srt_segments, 0.0);
            log.info(std::string("Wrote aligned JSON: ") + args.json_output.string());
          }
        } else {
          write_srt_utf8(args.output, srt_segments);
          log.info(std::string("Wrote aligned SRT: ") + args.output.string());
        }

      if (args.debug && !args.debug_dir.empty()) {
        fs::create_directories(args.debug_dir);

        // 01_original_segments.json
        {
          std::ofstream f(args.debug_dir / "01_original_segments.json", std::ios::binary);
          // Match align.py: dump original segments (before alignment) as a plain array of objects.
          f << "[\n";
          for (size_t i = 0; i < original_segments_for_debug.size(); ++i) {
            const auto& seg = original_segments_for_debug[i];
            f << "  {\"index\": " << (i + 1) << ", \"start\": " << seg.start_sec << ", \"end\": " << seg.end_sec
              << ", \"text\": " << json_quote_str(seg.text) << ", \"score\": " << seg.score << "}";
            if (i + 1 != original_segments_for_debug.size()) f << ",";
            f << "\n";
          }
          f << "]\n";
        }
        // 02_full_text.txt
        {
          std::ofstream f(args.debug_dir / "02_full_text.txt", std::ios::binary);
          f.write(full_text_for_debug.data(), std::streamsize(full_text_for_debug.size()));
        }

        auto write_json_string_array = [](const fs::path& p, const std::vector<std::string>& arr) {
          std::ofstream f(p, std::ios::binary);
          f << "[\n";
          for (size_t i = 0; i < arr.size(); ++i) {
            std::string e;
            e.reserve(arr[i].size() + 8);
            for (char c : arr[i]) {
              switch (c) {
                case '\\':
                  e += "\\\\";
                  break;
                case '"':
                  e += "\\\"";
                  break;
                case '\n':
                  e += "\\n";
                  break;
                case '\r':
                  e += "\\r";
                  break;
                case '\t':
                  e += "\\t";
                  break;
                default:
                  e.push_back(c);
                  break;
              }
            }
            f << "  \"" << e << "\"";
            if (i + 1 != arr.size()) f << ",";
            f << "\n";
          }
          f << "]\n";
        };

        write_json_string_array(args.debug_dir / "03_tokens_starred.json", tokens_starred);
        write_json_string_array(args.debug_dir / "04_text_starred.json", text_starred);

        // 05_word_timestamps.json (list of {text,start,end,score})
        {
          std::ofstream f(args.debug_dir / "05_word_timestamps.json", std::ios::binary);
          f << "[\n";
          for (size_t i = 0; i < word_ts.size(); ++i) {
            const auto& w = word_ts[i];
            f << "  {\"text\": " << json_quote_str(w.text) << ", \"start\": " << w.start_sec << ", \"end\": " << w.end_sec
              << ", \"score\": " << w.score << "}";
            if (i + 1 != word_ts.size()) f << ",";
            f << "\n";
          }
          f << "]\n";
        }

        // 06_aligned_segments.json (list of {index,start,end,text,score})
        {
          std::ofstream f(args.debug_dir / "06_aligned_segments.json", std::ios::binary);
          f << "[\n";
          for (size_t i = 0; i < srt_segments.size(); ++i) {
            const auto& seg = srt_segments[i];
            f << "  {\"index\": " << (i + 1) << ", \"start\": " << seg.start_sec << ", \"end\": " << seg.end_sec
              << ", \"text\": " << json_quote_str(seg.text) << ", \"score\": " << seg.score << "}";
            if (i + 1 != srt_segments.size()) f << ",";
            f << "\n";
          }
          f << "]\n";
        }

        // 00_summary.json (match align.py keys)
        {
          std::ofstream f(args.debug_dir / "00_summary.json", std::ios::binary);
          f << "{\n";
          f << "  \"audio_path\": " << json_quote_str(args.audio.string()) << ",\n";
          f << "  \"srt_path\": " << json_quote_str(args.srt.string()) << ",\n";
          f << "  \"language\": " << json_quote_str(language) << ",\n";
          f << "  \"romanize\": " << (romanize ? "true" : "false") << ",\n";
          f << "  \"audio_duration\": " << (audio_samples.size() / 16000.0) << ",\n";
          f << "  \"num_segments\": " << srt_segments.size() << ",\n";
          f << "  \"num_words\": " << word_ts.size() << ",\n";
          // align.py stores processing_time in seconds; we keep 0.0 for now (can be wired later if needed).
          f << "  \"processing_time\": " << 0.0 << "\n";
          f << "}\n";
        }
      }
  } catch (const std::exception& e) {
    log.error(std::string("Alignment failed: ") + e.what());
    throw;  // Re-throw to be caught by top-level handler with stack trace
  }

  return 0;
}
