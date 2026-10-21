#pragma once

#include "../interface.hpp"
#include "../parser.hpp"

namespace ae::lang::programing {

class Parser final : public ::ae::lang::Parser {
public:
    Parser() noexcept;

private:
    static ::ae::lang::ParseResult parse_callback(
        void* context,
        std::string_view text);
    ::ae::lang::ParseResult parse_impl(std::string_view text) const;
};

} // namespace ae::lang::programing
