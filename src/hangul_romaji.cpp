#include "hangul_romaji.h"

#include <cstdint>

namespace hangul {

namespace {

// Get the Unicode code point from a UTF-8 string_view (first character only)
uint32_t utf8_to_codepoint(std::string_view s) {
    if (s.empty()) return 0;

    unsigned char c0 = static_cast<unsigned char>(s[0]);

    // 1-byte sequence (ASCII)
    if ((c0 & 0x80) == 0) {
        return c0;
    }
    // 2-byte sequence
    if ((c0 & 0xE0) == 0xC0 && s.size() >= 2) {
        unsigned char c1 = static_cast<unsigned char>(s[1]);
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    // 3-byte sequence (Hangul syllables are here: U+AC00 to U+D7A3)
    if ((c0 & 0xF0) == 0xE0 && s.size() >= 3) {
        unsigned char c1 = static_cast<unsigned char>(s[1]);
        unsigned char c2 = static_cast<unsigned char>(s[2]);
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    // 4-byte sequence
    if ((c0 & 0xF8) == 0xF0 && s.size() >= 4) {
        unsigned char c1 = static_cast<unsigned char>(s[1]);
        unsigned char c2 = static_cast<unsigned char>(s[2]);
        unsigned char c3 = static_cast<unsigned char>(s[3]);
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }

    return 0;
}

// Get the byte length of a UTF-8 character from its first byte
size_t utf8_char_len(unsigned char first_byte) {
    if ((first_byte & 0x80) == 0) return 1;
    if ((first_byte & 0xE0) == 0xC0) return 2;
    if ((first_byte & 0xF0) == 0xE0) return 3;
    if ((first_byte & 0xF8) == 0xF0) return 4;
    return 1;  // Invalid, treat as 1
}

// Hangul Jamo romanization tables (from uroman)
// Initial consonants (choseong) - 19 jamo
const char* const LEADS[] = {
    "g", "gg", "n", "d", "dd", "r", "m", "b", "bb", "s",
    "ss", "", "j", "jj", "c", "k", "t", "p", "h"  // index 14: c (not ch)
};

// Medial vowels (jungseong) - 21 jamo
const char* const VOWELS[] = {
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa",
    "wai", "oe", "yo", "u", "weo", "we", "wi", "yu", "eu", "yi", "i"  // index 10: wai (not wae)
};

// Final consonants (jongseong) - 28 jamo (including none)
const char* const TAILS[] = {
    "", "g", "gg", "gs", "n", "nj", "nh", "d", "l", "lg",
    "lm", "lb", "ls", "lt", "lp", "lh", "m", "b", "bs", "s",
    "ss", "ng", "j", "c", "k", "t", "p", "h"  // index 23: c (not ch)
};

}  // anonymous namespace

bool is_hangul(std::string_view ch) {
    uint32_t cp = utf8_to_codepoint(ch);
    // Hangul Syllables: U+AC00 to U+D7A3
    return cp >= 0xAC00 && cp <= 0xD7A3;
}

std::string hangul_to_romaji(std::string_view hangul_char) {
    uint32_t cp = utf8_to_codepoint(hangul_char);

    // Check if it's a Hangul syllable
    if (cp < 0xAC00 || cp > 0xD7A3) {
        return "";
    }

    // Decompose the syllable
    // Hangul syllable = (lead * 21 + vowel) * 28 + tail + 0xAC00
    uint32_t code = cp - 0xAC00;
    int lead_index = static_cast<int>(code / (28 * 21));
    int vowel_index = static_cast<int>((code / 28) % 21);
    int tail_index = static_cast<int>(code % 28);

    std::string result;
    result += LEADS[lead_index];
    result += VOWELS[vowel_index];
    result += TAILS[tail_index];

    return result;
}

std::string romanize_hangul(const std::string& text) {
    std::string result;
    result.reserve(text.size() * 2);

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        size_t char_len = utf8_char_len(c);

        // Ensure we don't read past the end
        if (i + char_len > text.size()) {
            result += text[i];
            ++i;
            continue;
        }

        std::string_view char_view(text.data() + i, char_len);

        if (is_hangul(char_view)) {
            result += hangul_to_romaji(char_view);
        } else {
            // Not Hangul, keep as-is
            result.append(char_view);
        }
        i += char_len;
    }

    return result;
}

}  // namespace hangul
