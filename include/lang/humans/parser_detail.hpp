#pragma once

#include "parser.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <string_view>

namespace ae::lang::detail {

struct Affix {
    std::string_view text;
    MorphemeType type;
};

inline bool is_word_byte(unsigned char byte) noexcept {
    return byte >= 0x80 ||
           std::isalnum(byte) != 0 ||
           byte == '\'';
}

inline std::string_view longest_affix(
    std::string_view word,
    const auto& affixes,
    bool at_start) noexcept {
    std::string_view match;
    for (const Affix& affix : affixes) {
        const bool matches = at_start
            ? word.starts_with(affix.text)
            : word.ends_with(affix.text);
        if (matches && affix.text.size() > match.size()) {
            match = affix.text;
        }
    }
    return match;
}

inline MorphemeType affix_type(std::string_view affix, const auto& affixes) noexcept {
    for (const Affix& candidate : affixes) {
        if (candidate.text == affix) {
            return candidate.type;
        }
    }
    return MorphemeType::SUFFIX;
}

inline void append_morpheme(
    ParseResult& result,
    std::string_view text,
    MorphemeType type,
    std::size_t begin) {
    if (!text.empty()) {
        result.morphemes.push_back(
            Morpheme{std::string(text), type, begin, begin + text.size()});
    }
}

template <std::size_t PrefixCount, std::size_t SuffixCount>
ParseResult parse_words(
    std::string_view text,
    Language language,
    const std::array<Affix, PrefixCount>& prefixes,
    const std::array<Affix, SuffixCount>& suffixes) {
    ParseResult result{};
    result.language = language;
    std::size_t cursor = 0;

    while (cursor < text.size()) {
        while (cursor < text.size() &&
               !is_word_byte(static_cast<unsigned char>(text[cursor]))) {
            ++cursor;
        }
        const std::size_t word_begin = cursor;
        while (cursor < text.size() &&
               is_word_byte(static_cast<unsigned char>(text[cursor]))) {
            ++cursor;
        }
        if (word_begin == cursor) {
            continue;
        }

        std::string_view word = text.substr(word_begin, cursor - word_begin);
        std::size_t offset = word_begin;
        while (!word.empty()) {
            const std::string_view prefix = longest_affix(word, prefixes, true);
            if (prefix.empty() || prefix.size() == word.size()) {
                break;
            }
            append_morpheme(result, prefix, MorphemeType::PREFIX, offset);
            word.remove_prefix(prefix.size());
            offset += prefix.size();
        }

        std::string_view suffix;
        for (const Affix& candidate : suffixes) {
            if (word.ends_with(candidate.text) &&
                candidate.text.size() > suffix.size() &&
                candidate.text.size() < word.size()) {
                suffix = candidate.text;
            }
        }
        const std::size_t root_size = word.size() - suffix.size();
        append_morpheme(result, word.substr(0, root_size), MorphemeType::ROOT, offset);
        if (!suffix.empty()) {
            append_morpheme(
                result, suffix, affix_type(suffix, suffixes), offset + root_size);
        }
    }
    return result;
}

} // namespace ae::lang::detail
