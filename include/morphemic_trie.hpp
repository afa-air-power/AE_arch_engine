#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable)
// Module 3.2: Morphemic Tokenizer Compiler (Patricia Trie + 64-bit Boolean Mask)
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ae::mind1 {

struct MorphemicPatriciaTrie {
    enum class MorphemeType : std::uint8_t {
        PREFIX = 0,
        ROOT = 1,
        SUFFIX = 2,
        INFLECTION = 3
    };

    struct Node {
        std::string edge_label;
        int morpheme_id = -1;
        MorphemeType type = MorphemeType::ROOT;
        std::uint64_t compatibility_mask = 0;
        std::vector<std::pair<char, int>> children;
    };

    std::vector<Node> nodes;
    std::vector<std::string> dictionary;

    MorphemicPatriciaTrie() {
        nodes.push_back(Node{"", -1, MorphemeType::ROOT, 0, {}});
        insert("пере", 0, MorphemeType::PREFIX, 0x0000FFFFFFFFFFFULL);
        insert("под", 1, MorphemeType::PREFIX, 0x0000FFFFFFFFFFFULL);
        insert("раз", 2, MorphemeType::PREFIX, 0x0000FFFFFFFFFFFULL);
        insert("граф", 3, MorphemeType::ROOT, 0x00010000000001FULL);
        insert("мысл", 4, MorphemeType::ROOT, 0x00020000000003FULL);
        insert("знан", 5, MorphemeType::ROOT, 0x00040000000007FULL);
        insert("поток", 6, MorphemeType::ROOT, 0x0008000000000FFULL);
        insert("дерев", 7, MorphemeType::ROOT, 0x0010000000001FFULL);
        insert("корен", 8, MorphemeType::ROOT, 0x0020000000003FFULL);
        insert("центр", 9, MorphemeType::ROOT, 0x0040000000007FFULL);
        insert("память", 10, MorphemeType::ROOT, 0x008000000000FFFULL);
        insert("вектор", 11, MorphemeType::ROOT, 0x010000000001FFFULL);
        insert("токен", 12, MorphemeType::ROOT, 0x020000000003FFFULL);
        insert("модел", 13, MorphemeType::ROOT, 0x040000000007FFFULL);
        insert("индекс", 14, MorphemeType::ROOT, 0x08000000000FFFFULL);
        insert("аст", 15, MorphemeType::ROOT, 0x10000000001FFFFULL);
        insert("ов", 16, MorphemeType::SUFFIX, 0xFFFFFFFFFFFFFFFULL);
        insert("ени", 17, MorphemeType::SUFFIX, 0xFFFFFFFFFFFFFFFULL);
        insert("аци", 18, MorphemeType::SUFFIX, 0xFFFFFFFFFFFFFFFULL);
        insert("ость", 19, MorphemeType::SUFFIX, 0xFFFFFFFFFFFFFFFULL);
        insert("ый", 20, MorphemeType::INFLECTION, 0xFFFFFFFFFFFFFFFULL);
        insert("ам", 21, MorphemeType::INFLECTION, 0xFFFFFFFFFFFFFFFULL);
    }

    int find_child(int u, char c) const noexcept {
        for (const auto& [ch, nxt] : nodes[u].children) {
            if (ch == c) return nxt;
        }
        return -1;
    }

    void insert(std::string_view word, int id, MorphemeType type, std::uint64_t mask) {
        if (dictionary.size() <= static_cast<std::size_t>(id)) {
            dictionary.resize(id + 1);
        }
        dictionary[id] = std::string(word);

        int cur = 0;
        std::size_t pos = 0;
        while (pos < word.size()) {
            char c = word[pos];
            int nxt = find_child(cur, c);
            if (nxt == -1) {
                int new_idx = static_cast<int>(nodes.size());
                nodes.push_back(Node{std::string(word.substr(pos)), id, type, mask, {}});
                nodes[cur].children.push_back({c, new_idx});
                return;
            }
            std::string_view label = nodes[nxt].edge_label;
            std::size_t common = 0;
            while (common < label.size() && pos + common < word.size() &&
                   label[common] == word[pos + common]) {
                ++common;
            }
            if (common == label.size()) {
                cur = nxt;
                pos += common;
            } else {
                int split_idx = static_cast<int>(nodes.size());
                Node split_node{std::string(label.substr(0, common)), -1, MorphemeType::ROOT, 0, {}};
                nodes[nxt].edge_label = std::string(label.substr(common));
                split_node.children.push_back({nodes[nxt].edge_label[0], nxt});

                for (auto& [ch, child_idx] : nodes[cur].children) {
                    if (child_idx == nxt) {
                        child_idx = split_idx;
                        break;
                    }
                }
                nodes.push_back(std::move(split_node));

                pos += common;
                if (pos == word.size()) {
                    nodes[split_idx].morpheme_id = id;
                    nodes[split_idx].type = type;
                    nodes[split_idx].compatibility_mask = mask;
                } else {
                    int leaf_idx = static_cast<int>(nodes.size());
                    nodes.push_back(Node{std::string(word.substr(pos)), id, type, mask, {}});
                    nodes[split_idx].children.push_back({word[pos], leaf_idx});
                }
                return;
            }
        }
        nodes[cur].morpheme_id = id;
        nodes[cur].type = type;
        nodes[cur].compatibility_mask = mask;
    }

    __attribute__((noinline)) std::pair<std::uint64_t, int> compile_morpheme_mask(
        std::string_view text, const std::array<float, 16>& intent) const noexcept {
        std::uint64_t mask = 0x00000000000000FFULL;
        int matched_morphemes = 0;

        for (std::size_t i = 0; i < text.size(); ++i) {
            int cur = 0;
            std::size_t pos = i;
            while (pos < text.size()) {
                int nxt = find_child(cur, text[pos]);
                if (nxt == -1) break;
                std::string_view label = nodes[nxt].edge_label;
                if (text.substr(pos).starts_with(label)) {
                    pos += label.size();
                    cur = nxt;
                    if (nodes[cur].morpheme_id != -1) {
                        mask |= nodes[cur].compatibility_mask;
                        ++matched_morphemes;
                    }
                } else {
                    break;
                }
            }
        }

        for (std::size_t b = 0; b < 16; ++b) {
            if (intent[b] > -0.25f) {
                mask |= (1ULL << (b + 8));
            }
        }
        return {mask, matched_morphemes};
    }
};

} // namespace ae::mind1
