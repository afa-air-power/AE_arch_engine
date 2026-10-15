#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable)
// Module 3.4: Graph-Decoder Stream (Zero-Allocation Cache-Local Dinic Flow)
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace ae::mind {

struct DinitzGraphDecoder {
    static constexpr int MAX_V = 128;
    static constexpr int MAX_E = 1024;

    struct FlatEdge {
        int to;
        int next;
        int base_cap;
        int cap;
        int flow;
        std::uint8_t semantic_tag;
    };

    int n = 0;
    int edge_count = 0;
    alignas(64) std::array<int, MAX_V> head{};
    alignas(64) std::array<int, MAX_V> level{};
    alignas(64) std::array<int, MAX_V> ptr{};
    alignas(64) std::array<int, MAX_V> q_buf{};
    alignas(64) std::array<FlatEdge, MAX_E> edges{};

    explicit DinitzGraphDecoder(int nodes = 64) : n(nodes), edge_count(0) {
        head.fill(-1);
    }

    inline void add_static_edge(int from, int to, int base_cap, std::uint8_t tag) noexcept {
        if (edge_count + 2 > MAX_E) return;
        edges[edge_count] = {to, head[from], base_cap, base_cap, 0, tag};
        head[from] = edge_count++;
        edges[edge_count] = {from, head[to], 0, 0, 0, tag};
        head[to] = edge_count++;
    }

    inline void apply_boolean_mask_and_reset(std::uint64_t boolean_mask) noexcept {
        for (int e = 0; e < edge_count; e += 2) {
            std::uint8_t tag = edges[e].semantic_tag;
            int gated = ((boolean_mask >> (tag & 63)) & 1ULL) ? edges[e].base_cap : 0;
            edges[e].cap = gated;
            edges[e].flow = 0;
            edges[e ^ 1].cap = 0;
            edges[e ^ 1].flow = 0;
        }
    }

    inline bool bfs(int s, int t) noexcept {
        for (int i = 0; i < n; ++i) level[i] = -1;
        level[s] = 0;
        int qh = 0, qt = 0;
        q_buf[qt++] = s;
        while (qh < qt) {
            int v = q_buf[qh++];
            for (int id = head[v]; id != -1; id = edges[id].next) {
                int tr = edges[id].to;
                if (edges[id].cap - edges[id].flow > 0 && level[tr] == -1) {
                    level[tr] = level[v] + 1;
                    q_buf[qt++] = tr;
                }
            }
        }
        return level[t] != -1;
    }

    inline int dfs(int v, int t, int pushed) noexcept {
        if (pushed == 0 || v == t) return pushed;
        for (int& cid = ptr[v]; cid != -1; cid = edges[cid].next) {
            int id = cid;
            int tr = edges[id].to;
            if (level[v] + 1 != level[tr] || edges[id].cap - edges[id].flow == 0) continue;
            int tr_pushed = dfs(tr, t, std::min(pushed, edges[id].cap - edges[id].flow));
            if (tr_pushed == 0) continue;
            edges[id].flow += tr_pushed;
            edges[id ^ 1].flow -= tr_pushed;
            return tr_pushed;
        }
        return 0;
    }

    inline int max_flow(int s, int t) noexcept {
        int flow = 0;
        while (bfs(s, t)) {
            for (int i = 0; i < n; ++i) ptr[i] = head[i];
            while (int pushed = dfs(s, t, 1000000000)) {
                flow += pushed;
            }
        }
        return flow;
    }

    inline void extract_optimal_thought_path(int s, int t, std::vector<int>& path) const noexcept {
        path.clear();
        path.push_back(s);
        std::array<std::uint8_t, MAX_V> vis{};
        int cur = s;
        vis[s] = 1;
        while (cur != t) {
            int best_next = -1;
            int best_flow = 0;
            for (int id = head[cur]; id != -1; id = edges[id].next) {
                if ((id & 1) == 0 && edges[id].flow > best_flow && !vis[edges[id].to]) {
                    best_flow = edges[id].flow;
                    best_next = edges[id].to;
                }
            }
            if (best_next == -1) break;
            vis[best_next] = 1;
            path.push_back(best_next);
            cur = best_next;
        }
    }
};

} // namespace ae::mind1
