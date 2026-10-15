#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ae::lang {

enum class Language {
    RUSSIAN,
    ENGLISH
};

enum class MorphemeType {
    PREFIX,
    ROOT,
    SUFFIX,
    INFLECTION
};

struct Morpheme {
    std::string text;
    MorphemeType type = MorphemeType::ROOT;
    std::size_t begin = 0;
    std::size_t end = 0;
};

struct ParseResult {
    Language language = Language::ENGLISH;
    std::vector<Morpheme> morphemes;
};

class Parser {
public:
    using ParseCallback = ParseResult (*)(void*, std::string_view);

    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
    Parser(Parser&&) = delete;
    Parser& operator=(Parser&&) = delete;
    ~Parser() = default;

    ParseResult parse(std::string_view text) const;
    Language language() const noexcept;

protected:
    Parser(void* context, ParseCallback callback, Language language) noexcept;

private:
    void* context_;
    ParseCallback callback_;
    Language language_;
};

Language detect_language(std::string_view text) noexcept;
Parser* select_parser(std::string_view text) noexcept;

} // namespace ae::lang
