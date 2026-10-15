#include "../../../lang/humans/en/parser.hpp"

#include "../../../lang/humans/parser_detail.hpp"

#include <array>

namespace ae::lang::en {
namespace {

constexpr std::array prefixes{
    detail::Affix{"under", MorphemeType::PREFIX},
    detail::Affix{"over", MorphemeType::PREFIX},
    detail::Affix{"inter", MorphemeType::PREFIX},
    detail::Affix{"dis", MorphemeType::PREFIX},
    detail::Affix{"pre", MorphemeType::PREFIX},
    detail::Affix{"mis", MorphemeType::PREFIX},
    detail::Affix{"re", MorphemeType::PREFIX},
    detail::Affix{"un", MorphemeType::PREFIX},
    detail::Affix{"non", MorphemeType::PREFIX},
};

constexpr std::array suffixes{
    detail::Affix{"ization", MorphemeType::SUFFIX},
    detail::Affix{"ational", MorphemeType::SUFFIX},
    detail::Affix{"ness", MorphemeType::SUFFIX},
    detail::Affix{"ment", MorphemeType::SUFFIX},
    detail::Affix{"tion", MorphemeType::SUFFIX},
    detail::Affix{"ing", MorphemeType::INFLECTION},
    detail::Affix{"ied", MorphemeType::INFLECTION},
    detail::Affix{"ed", MorphemeType::INFLECTION},
    detail::Affix{"ies", MorphemeType::INFLECTION},
    detail::Affix{"es", MorphemeType::INFLECTION},
    detail::Affix{"s", MorphemeType::INFLECTION},
    detail::Affix{"ly", MorphemeType::SUFFIX},
    detail::Affix{"er", MorphemeType::INFLECTION},
    detail::Affix{"est", MorphemeType::INFLECTION},
};

} // namespace

Parser::Parser() noexcept
    : ae::lang::Parser(this, &Parser::parse_callback, Language::ENGLISH) {}

ParseResult Parser::parse_callback(void* context, std::string_view text) {
    return static_cast<Parser*>(context)->parse_impl(text);
}

ParseResult Parser::parse_impl(std::string_view text) const {
    return detail::parse_words(text, Language::ENGLISH, prefixes, suffixes);
}

} // namespace ae::lang::en
