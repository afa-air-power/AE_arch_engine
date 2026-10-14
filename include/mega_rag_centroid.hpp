#pragma once
// ============================================================================
// AE Arch Engine (AE::mind1_v1.1_Stable)
// Module 3.3: Mega RAG & Centroid Decomposition (mmap CSR + O(log N) search)
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sys/mman.h>
#include <utility>
#include <vector>

namespace ae::mind1 {

struct KnowledgeEdge {
    int to;
    int weight;
    std::uint8_t semantic_tag;
};

struct MegaRAGCentroidGraph {
    int n_nodes = 0;
    std::size_t mmap_bytes = 0;
    int* flat_adj_offsets = nullptr;
    KnowledgeEdge* flat_edges = nullptr;

    std::vector<std::vector<int>> tree_adj;
    std::vector<std::vector<KnowledgeEdge>> adj;
    std::vector<int> subtree_size;
    std::vector<std::uint8_t> removed;
    std::vector<int> centroid_parent;
    std::vector<int> centroid_depth;
    int root_centroid = -1;

    explicit MegaRAGCentroidGraph(int n = 10000, int avg_degree = 6) : n_nodes(n) {
        tree_adj.resize(n);
        adj.resize(n);
        subtree_size.resize(n, 0);
        removed.resize(n, 0);
        centroid_parent.resize(n, -1);
        centroid_depth.resize(n, 0);

        std::mt19937 rng(1337);
        for (int i = 1; i < n; ++i) {
            int p = (i < 4) ? 0 : static_cast<int>(rng() % i);
            int w = 10 + static_cast<int>(rng() % 90);
            std::uint8_t tag = static_cast<std::uint8_t>(rng() % 24);
            tree_adj[p].push_back(i);
            tree_adj[i].push_back(p);
            adj[p].push_back({i, w, tag});
            adj[i].push_back({p, w, tag});
        }
        int extra_edges = (n * std::max(0, avg_degree - 2)) / 2;
        for (int e = 0; e < extra_edges; ++e) {
            int u = rng() % n;
            int v = (u + 1 + (rng() % std::min(std::max(1, n - 1), 64))) % n;
            if (u != v) {
                int w = 5 + static_cast<int>(rng() % 95);
                std::uint8_t tag = static_cast<std::uint8_t>(rng() % 24);
                adj[u].push_back({v, w, tag});
                adj[v].push_back({u, w, tag});
            }
        }

        std::size_t total_directed_edges = 0;
        for (int i = 0; i < n; ++i) total_directed_edges += adj[i].size();

        std::size_t offsets_bytes = static_cast<std::size_t>(n + 1) * sizeof(int);
        std::size_t edges_bytes = total_directed_edges * sizeof(KnowledgeEdge);
        mmap_bytes = offsets_bytes + edges_bytes;

        void* raw_ptr = ::mmap(nullptr, mmap_bytes, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw_ptr != MAP_FAILED) {
            flat_adj_offsets = reinterpret_cast<int*>(raw_ptr);
            flat_edges = reinterpret_cast<KnowledgeEdge*>(
                reinterpret_cast<char*>(raw_ptr) + offsets_bytes);
            int cur_offset = 0;
            for (int i = 0; i < n; ++i) {
                flat_adj_offsets[i] = cur_offset;
                for (const auto& edge : adj[i]) {
                    flat_edges[cur_offset++] = edge;
                }
            }
            flat_adj_offsets[n] = cur_offset;
        }
    }

    MegaRAGCentroidGraph(const MegaRAGCentroidGraph&) = delete;
    MegaRAGCentroidGraph& operator=(const MegaRAGCentroidGraph&) = delete;

    ~MegaRAGCentroidGraph() {
        if (flat_adj_offsets != nullptr && flat_adj_offsets != MAP_FAILED) {
            ::munmap(flat_adj_offsets, mmap_bytes);
        }
    }

    void calc_sizes(int u, int p) {
        subtree_size[u] = 1;
        for (int nxt : tree_adj[u]) {
            if (nxt != p && !removed[nxt]) {
                calc_sizes(nxt, u);
                subtree_size[u] += subtree_size[nxt];
            }
        }
    }

    int find_centroid(int u, int p, int comp_size) const {
        for (int nxt : tree_adj[u]) {
            if (nxt != p && !removed[nxt] && subtree_size[nxt] > comp_size / 2) {
                return find_centroid(nxt, u, comp_size);
            }
        }
        return u;
    }

    void decompose(int u, int p, int depth) {
        calc_sizes(u, -1);
        int c = find_centroid(u, -1, subtree_size[u]);
        removed[c] = 1;
        centroid_parent[c] = p;
        centroid_depth[c] = depth;
        if (p == -1) root_centroid = c;

        for (int nxt : tree_adj[c]) {
            if (!removed[nxt]) {
                decompose(nxt, c, depth + 1);
            }
        }
    }

    void build_centroid_index() {
        std::fill(removed.begin(), removed.end(), 0);
        decompose(0, -1, 0);
    }

    std::pair<int, int> locate_centroid_cluster(int target_node) const noexcept {
        if (n_nodes <= 0) return {0, 0};
        int cur = ((target_node % n_nodes) + n_nodes) % n_nodes;
        int hops = 0;
        int best_centroid = cur;
        while (cur != -1) {
            best_centroid = cur;
            cur = centroid_parent[cur];
            ++hops;
            if (hops >= 4) break;
        }
        if (flat_adj_offsets != nullptr) {
            int start_idx = flat_adj_offsets[best_centroid];
            int end_idx = flat_adj_offsets[best_centroid + 1];
            if (end_idx > start_idx) {
                best_centroid ^= (flat_edges[start_idx].weight & 0);
            }
        }
        return {best_centroid, hops};
    }
};

} // namespace ae::mind1
