#include "../../../lang/humans/ru/parser.hpp"

#include "../../../lang/humans/parser_detail.hpp"

#include <array>

namespace ae::lang::ru {
namespace {

constexpr std::array prefixes{
    detail::Affix{"пере", MorphemeType::PREFIX},
    detail::Affix{"под", MorphemeType::PREFIX},
    detail::Affix{"раз", MorphemeType::PREFIX},
    detail::Affix{"при", MorphemeType::PREFIX},
    detail::Affix{"про", MorphemeType::PREFIX},
    detail::Affix{"от", MorphemeType::PREFIX},
    detail::Affix{"за", MorphemeType::PREFIX},
    detail::Affix{"на", MorphemeType::PREFIX},
    detail::Affix{"по", MorphemeType::PREFIX},
    detail::Affix{"не", MorphemeType::PREFIX},
};

constexpr std::array suffixes{
    detail::Affix{"ирование", MorphemeType::SUFFIX},
    detail::Affix{"енность", MorphemeType::SUFFIX},
    detail::Affix{"ениями", MorphemeType::INFLECTION},
    detail::Affix{"ение", MorphemeType::SUFFIX},
    detail::Affix{"ания", MorphemeType::SUFFIX},
    detail::Affix{"ация", MorphemeType::SUFFIX},
    detail::Affix{"ость", MorphemeType::SUFFIX},
    detail::Affix{"овать", MorphemeType::SUFFIX},
    detail::Affix{"ами", MorphemeType::INFLECTION},
    detail::Affix{"ями", MorphemeType::INFLECTION},
    detail::Affix{"ого", MorphemeType::INFLECTION},
    detail::Affix{"ему", MorphemeType::INFLECTION},
    detail::Affix{"ыми", MorphemeType::INFLECTION},
    detail::Affix{"ый", MorphemeType::INFLECTION},
    detail::Affix{"ая", MorphemeType::INFLECTION},
    detail::Affix{"ое", MorphemeType::INFLECTION},
    detail::Affix{"ые", MorphemeType::INFLECTION},
    detail::Affix{"ам", MorphemeType::INFLECTION},
    detail::Affix{"ах", MorphemeType::INFLECTION},
    detail::Affix{"ом", MorphemeType::INFLECTION},
    detail::Affix{"а", MorphemeType::INFLECTION},
    detail::Affix{"я", MorphemeType::INFLECTION},
    detail::Affix{"ы", MorphemeType::INFLECTION},
    detail::Affix{"и", MorphemeType::INFLECTION},
    detail::Affix{"у", MorphemeType::INFLECTION},
    detail::Affix{"е", MorphemeType::INFLECTION},
};

} // namespace

Parser::Parser() noexcept
    : ae::lang::Parser(this, &Parser::parse_callback, Language::RUSSIAN) {}

ParseResult Parser::parse_callback(void* context, std::string_view text) {
    return static_cast<Parser*>(context)->parse_impl(text);
}

ParseResult Parser::parse_impl(std::string_view text) const {
    return detail::parse_words(text, Language::RUSSIAN, prefixes, suffixes);
}

} // namespace ae::lang::ru
