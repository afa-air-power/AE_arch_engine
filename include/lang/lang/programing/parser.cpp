#include "../include/lang/programing/parser.hpp"

#include <array>
#include <cctype>
#include <string_view>
#include <utility>

namespace ae::lang::programing {
namespace {

struct Token {
    std::string_view text;
    ::ae::lang::MeaningKind kind;
    std::size_t begin;
    std::size_t end;
};

constexpr std::string_view keywords[]{
    "alignas", "auto", "bool", "break", "case", "catch",
    "char", "class", "const", "constexpr", "continue", "default", "delete",
    "do", "double", "else", "enum", "explicit", "extern", "false", "float",
    "for", "friend", "if", "inline", "int", "long", "namespace", "new",
    "nullptr", "private", "protected", "public", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "template", "this", "throw",
    "true", "try", "typedef", "typename", "union", "unsigned", "using",
    "virtual", "void", "volatile", "while"
};

bool is_identifier_start(unsigned char ch) noexcept {
    return ch == '_' || std::isalpha(ch) != 0 || ch >= 0x80;
}

bool is_identifier_byte(unsigned char ch) noexcept {
    return is_identifier_start(ch) || std::isdigit(ch) != 0;
}

bool is_keyword(std::string_view text) noexcept {
    for (std::string_view keyword : keywords) {
        if (text == keyword) {
            return true;
        }
    }
    return false;
}

bool is_control_keyword(std::string_view text) noexcept {
    return text == "if" || text == "else" || text == "for" ||
           text == "while" || text == "switch" || text == "case" ||
           text == "try" || text == "catch" || text == "do";
}

bool is_declaration_keyword(std::string_view text) noexcept {
    return text == "auto" || text == "bool" || text == "char" ||
           text == "class" || text == "const" || text == "constexpr" ||
           text == "double" || text == "enum" || text == "float" ||
           text == "int" || text == "long" || text == "namespace" ||
           text == "short" || text == "static" || text == "struct" ||
           text == "template" || text == "typedef" || text == "typename" ||
           text == "union" || text == "using" || text == "void";
}

std::vector<Token> tokenize(std::string_view source) {
    std::vector<Token> tokens;
    std::size_t cursor = 0;
    while (cursor < source.size()) {
        const unsigned char ch = static_cast<unsigned char>(source[cursor]);
        if (std::isspace(ch) != 0) {
            ++cursor;
            continue;
        }
        if (source.substr(cursor).starts_with("//")) {
            const std::size_t newline = source.find('\n', cursor + 2);
            cursor = newline == std::string_view::npos ? source.size() : newline + 1;
            continue;
        }
        if (source.substr(cursor).starts_with("/*")) {
            const std::size_t close = source.find("*/", cursor + 2);
            cursor = close == std::string_view::npos ? source.size() : close + 2;
            continue;
        }

        const std::size_t begin = cursor;
        ::ae::lang::MeaningKind kind = ::ae::lang::MeaningKind::PUNCTUATION;
        if (is_identifier_start(ch)) {
            ++cursor;
            while (cursor < source.size() &&
                   is_identifier_byte(static_cast<unsigned char>(source[cursor]))) {
                ++cursor;
            }
            const std::string_view word = source.substr(begin, cursor - begin);
            kind = is_keyword(word)
                ? ::ae::lang::MeaningKind::KEYWORD
                : ::ae::lang::MeaningKind::IDENTIFIER;
        } else if (std::isdigit(ch) != 0) {
            ++cursor;
            while (cursor < source.size()) {
                const unsigned char next = static_cast<unsigned char>(source[cursor]);
                if (std::isalnum(next) == 0 && next != '.' && next != '_') {
                    break;
                }
                ++cursor;
            }
            kind = ::ae::lang::MeaningKind::LITERAL;
        } else if (ch == '"' || ch == '\'') {
            const char quote = static_cast<char>(ch);
            ++cursor;
            while (cursor < source.size()) {
                if (source[cursor] == '\\' && cursor + 1 < source.size()) {
                    cursor += 2;
                } else if (source[cursor++] == quote) {
                    break;
                }
            }
            kind = ::ae::lang::MeaningKind::LITERAL;
        } else if (std::string_view("+-*/%=!<>|&^~?:").find(static_cast<char>(ch)) !=
                   std::string_view::npos) {
            ++cursor;
            while (cursor < source.size() &&
                   std::string_view("=<>|&").find(source[cursor]) !=
                       std::string_view::npos &&
                   cursor - begin < 2) {
                ++cursor;
            }
            kind = ::ae::lang::MeaningKind::OPERATOR;
        } else {
            ++cursor;
        }
        tokens.push_back(Token{source.substr(begin, cursor - begin), kind, begin, cursor});
    }
    return tokens;
}

::ae::lang::MeaningKind classify_statement(
    const std::vector<Token>& tokens,
    bool has_block) noexcept {
    if (tokens.empty()) {
        return ::ae::lang::MeaningKind::BLOCK;
    }
    if (is_control_keyword(tokens.front().text)) {
        return ::ae::lang::MeaningKind::CONTROL_FLOW;
    }
    if (has_block) {
        for (std::size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i].text == "(" && tokens[i - 1].kind ==
                    ::ae::lang::MeaningKind::IDENTIFIER) {
                return ::ae::lang::MeaningKind::FUNCTION;
            }
        }
    }
    for (const Token& token : tokens) {
        if (token.kind == ::ae::lang::MeaningKind::KEYWORD &&
            is_declaration_keyword(token.text)) {
            return ::ae::lang::MeaningKind::DECLARATION;
        }
    }
    return ::ae::lang::MeaningKind::STATEMENT;
}

std::string join_tokens(const std::vector<Token>& tokens) {
    std::string text;
    for (const Token& token : tokens) {
        if (!text.empty()) {
            text.push_back(' ');
        }
        text.append(token.text);
    }
    return text;
}

class TreeBuilder {
public:
    TreeBuilder(std::string_view source, const std::vector<Token>& tokens)
        : source_(source), tokens_(tokens) {}

    ::ae::lang::MeaningNode build() {
        ::ae::lang::MeaningNode root{
            ::ae::lang::MeaningKind::PROGRAM, {}, 0, source_.size(), {}};
        root.children = parse_scope();
        return root;
    }

private:
    std::vector<::ae::lang::MeaningNode> parse_scope() {
        std::vector<::ae::lang::MeaningNode> nodes;
        std::vector<Token> statement;
        while (index_ < tokens_.size()) {
            const Token& token = tokens_[index_];
            if (token.text == "}") {
                if (!statement.empty()) {
                    nodes.push_back(make_statement(statement, false));
                    statement.clear();
                }
                ++index_;
                break;
            }
            if (token.text == "{") {
                const std::size_t block_begin = token.begin;
                ++index_;
                ::ae::lang::MeaningNode block{
                    ::ae::lang::MeaningKind::BLOCK, {}, block_begin, block_begin, {}};
                block.children = parse_scope();
                block.end = index_ == 0 ? block_begin : tokens_[index_ - 1].end;
                if (statement.empty()) {
                    nodes.push_back(std::move(block));
                } else {
                    auto node = make_statement(statement, true);
                    node.children.push_back(std::move(block));
                    node.end = node.children.back().end;
                    nodes.push_back(std::move(node));
                    statement.clear();
                }
                continue;
            }
            if (token.text == ";") {
                statement.push_back(token);
                ++index_;
                nodes.push_back(make_statement(statement, false));
                statement.clear();
                continue;
            }
            statement.push_back(token);
            ++index_;
        }
        if (!statement.empty()) {
            nodes.push_back(make_statement(statement, false));
        }
        return nodes;
    }

    ::ae::lang::MeaningNode make_statement(
        const std::vector<Token>& tokens,
        bool has_block) const {
        ::ae::lang::MeaningNode node{
            classify_statement(tokens, has_block),
            join_tokens(tokens),
            tokens.front().begin,
            tokens.back().end,
            {}};
        node.children.reserve(tokens.size());
        for (const Token& token : tokens) {
            node.children.push_back(::ae::lang::MeaningNode{
                token.kind, std::string(token.text), token.begin, token.end, {}});
        }
        return node;
    }

    std::string_view source_;
    const std::vector<Token>& tokens_;
    std::size_t index_ = 0;
};

} // namespace

Parser::Parser() noexcept
    : ::ae::lang::Parser(
          this,
          &Parser::parse_callback,
          ::ae::lang::Language::PROGRAMMING,
          ::ae::lang::LanguageFamily::PROGRAMMING) {}

::ae::lang::ParseResult Parser::parse_callback(void* context, std::string_view text) {
    return static_cast<Parser*>(context)->parse_impl(text);
}

::ae::lang::ParseResult Parser::parse_impl(std::string_view text) const {
    const std::vector<Token> tokens = tokenize(text);
    TreeBuilder builder(text, tokens);
    ::ae::lang::ParseResult result{};
    result.language = ::ae::lang::Language::PROGRAMMING;
    result.family = ::ae::lang::LanguageFamily::PROGRAMMING;
    result.meaning_tree = builder.build();
    return result;
}

} // namespace ae::lang::programing
