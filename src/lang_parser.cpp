#include "../include/lang/humans/parser.hpp"

#include "../include/lang/humans/en/parser.hpp"
#include "../include/lang/programing/parser.hpp"
#include "../include/lang/humans/ru/parser.hpp"

namespace ae::lang {
namespace {

ru::Parser russian_parser;
en::Parser english_parser;
programing::Parser programming_parser;

} // namespace

Parser::Parser(
    void* context,
    ParseCallback callback,
    Language language,
    LanguageFamily family) noexcept
    : context_(context), callback_(callback), language_(language), family_(family) {}

ParseResult Parser::parse(std::string_view text) const {
    ParseResult result = callback_(context_, text);
    result.family = family_;
    return result;
}

Language Parser::language() const noexcept {
    return language_;
}

LanguageFamily Parser::family() const noexcept {
    return family_;
}

Language detect_language(std::string_view text) noexcept {
    std::size_t cyrillic_bytes = 0;
    std::size_t latin_letters = 0;
    for (unsigned char byte : text) {
        if (byte >= 0xD0 && byte <= 0xD1) {
            ++cyrillic_bytes;
        } else if ((byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z')) {
            ++latin_letters;
        }
    }
    return cyrillic_bytes > latin_letters
        ? Language::RUSSIAN
        : Language::ENGLISH;
}

Parser* select_parser(std::string_view text) noexcept {
    return detect_language(text) == Language::RUSSIAN
        ? static_cast<Parser*>(&russian_parser)
        : static_cast<Parser*>(&english_parser);
}

LanguageInterface* select_parser(
    LanguageFamily family,
    std::string_view text) noexcept {
    switch (family) {
    case LanguageFamily::HUMAN:
        return select_parser(text);
    case LanguageFamily::PROGRAMMING:
        return &programming_parser;
    case LanguageFamily::SCIENCE:
    case LanguageFamily::VERBAL:
        return nullptr;
    }
    return nullptr;
}

} // namespace ae::lang
