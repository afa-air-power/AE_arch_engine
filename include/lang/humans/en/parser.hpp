#pragma once

#include "../parser.hpp"

namespace ae::lang::en {

class Parser final : public ae::lang::Parser {
public:
    Parser() noexcept;

private:
    static ParseResult parse_callback(void* context, std::string_view text);
    ParseResult parse_impl(std::string_view text) const;
};

} // namespace ae::lang::en
