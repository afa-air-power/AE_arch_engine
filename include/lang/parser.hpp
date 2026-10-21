#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace ae::lang {

enum class Language {
    RUSSIAN,
    ENGLISH,
    PROGRAMMING
};

enum class LanguageFamily {
    HUMAN,
    SCIENCE,
    VERBAL,
    PROGRAMMING
};
using Domain = LanguageFamily;

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

enum class MeaningKind {
    DOCUMENT,
    PROGRAM,
    FUNCTION,
    DECLARATION,
    CONTROL_FLOW,
    BLOCK,
    STATEMENT,
    IDENTIFIER,
    KEYWORD,
    LITERAL,
    OPERATOR,
    PUNCTUATION
};

struct MeaningNode {
    MeaningKind kind = MeaningKind::DOCUMENT;
    std::string text;
    std::size_t begin = 0;
    std::size_t end = 0;
    std::vector<MeaningNode> children;
};

struct ParseResult {
    Language language = Language::ENGLISH;
    std::vector<Morpheme> morphemes;
    MeaningNode meaning_tree;
    LanguageFamily family = LanguageFamily::HUMAN;
};

class LanguageInterface {
public:
    virtual ~LanguageInterface() = default;
    virtual ParseResult parse(std::string_view text) const = 0;
    virtual Language language() const noexcept = 0;
    virtual LanguageFamily family() const noexcept = 0;
};

class Parser : public LanguageInterface {
public:
    using ParseCallback = ParseResult (*)(void*, std::string_view);

    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;
    Parser(Parser&&) = delete;
    Parser& operator=(Parser&&) = delete;
    ~Parser() override = default;

    [[nodiscard]] ParseResult parse(std::string_view text) const override;
    [[nodiscard]] Language language() const noexcept override;
    [[nodiscard]] LanguageFamily family() const noexcept override;

protected:
    Parser(
        void* context,
        ParseCallback callback,
        Language language,
        LanguageFamily family = LanguageFamily::HUMAN) noexcept;

private:
    void* context_;
    ParseCallback callback_;
    Language language_;
    LanguageFamily family_;
};

Language detect_language(std::string_view text) noexcept;
Parser* select_parser(std::string_view text) noexcept;
LanguageInterface* select_parser(
    LanguageFamily family,
    std::string_view text) noexcept;

} // namespace ae::lang
