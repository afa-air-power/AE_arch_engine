#pragma once
// ============================================================================
// AE Arch Engine (AE::mind_v1.1_Stable)
// Module 4: Thought Graph Topology (Kruskal DSU + BINUP LCA + Dynamic Reroot)
// Author: Andrey Firstov (IU5 BMSTU)
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

namespace ae::mind {

struct ThoughtGraphTopology {
    struct Edge {
        int u, v, weight, id;
        std::uint8_t tag;
        bool operator<(const Edge& other) const noexcept {
            return weight > other.weight;
        }
    };

    struct DSU {
        std::vector<int> parent;
        explicit DSU(int n) : parent(n) {
            std::iota(parent.begin(), parent.end(), 0);
        }
        int find(int i) noexcept {
            if (parent[i] == i) return i;
            return parent[i] = find(parent[i]);
        }
        bool unite(int i, int j) noexcept {
            int root_i = find(i);
            int root_j = find(j);
            if (root_i != root_j) {
                parent[root_i] = root_j;
                return true;
            }
            return false;
        }
    };

    int n = 0;
    int log_n = 1;
    int current_root = 0;
    std::vector<Edge> all_edges;
    std::vector<std::vector<std::pair<int, int>>> adj_mst;
    std::vector<std::vector<std::pair<int, int>>> adj_horiz;
    std::vector<std::vector<int>> up;
    std::vector<std::vector<int>> max_edge;
    std::vector<int> depth;

    void init_from_subgraph(int num_nodes, const std::vector<Edge>& edges) {
        n = num_nodes;
        log_n = 1;
        while ((1 << log_n) <= n) ++log_n;
        log_n = std::max(log_n, 8);

        all_edges = edges;
        adj_mst.assign(n, {});
        adj_horiz.assign(n, {});
        up.assign(n, std::vector<int>(log_n, 0));
        max_edge.assign(n, std::vector<int>(log_n, 0));
        depth.assign(n, 0);

        std::vector<Edge> sorted_edges = edges;
        std::sort(sorted_edges.begin(), sorted_edges.end());
        DSU dsu(n);

        for (const auto& e : sorted_edges) {
            if (dsu.unite(e.u, e.v)) {
                adj_mst[e.u].push_back({e.v, e.weight});
                adj_mst[e.v].push_back({e.u, e.weight});
            } else {
                adj_horiz[e.u].push_back({e.v, e.weight});
                adj_horiz[e.v].push_back({e.u, e.weight});
            }
        }

        for (int i = 1; i < n; ++i) {
            if (dsu.unite(0, i)) {
                adj_mst[0].push_back({i, 1});
                adj_mst[i].push_back({0, 1});
            }
        }

        current_root = 0;
        dfs_lca(current_root, current_root, 0, 0);
    }

    void dfs_lca(int u, int p, int w, int d) noexcept {
        depth[u] = d;
        up[u][0] = p;
        max_edge[u][0] = w;

        for (int i = 1; i < log_n; ++i) {
            int mid = up[u][i - 1];
            up[u][i] = up[mid][i - 1];
            max_edge[u][i] = std::max(max_edge[u][i - 1], max_edge[mid][i - 1]);
        }

        for (const auto& [v, weight] : adj_mst[u]) {
            if (v != p) {
                dfs_lca(v, u, weight, d + 1);
            }
        }
    }

    std::pair<int, int> query_path_binup(int u, int v) const noexcept {
        int res = 0;
        if (depth[u] < depth[v]) std::swap(u, v);

        for (int i = log_n - 1; i >= 0; --i) {
            if (depth[u] - (1 << i) >= depth[v]) {
                res = std::max(res, max_edge[u][i]);
                u = up[u][i];
            }
        }
        if (u == v) return {u, res};

        for (int i = log_n - 1; i >= 0; --i) {
            if (up[u][i] != up[v][i]) {
                res = std::max(res, max_edge[u][i]);
                res = std::max(res, max_edge[v][i]);
                u = up[u][i];
                v = up[v][i];
            }
        }
        res = std::max(res, max_edge[u][0]);
        res = std::max(res, max_edge[v][0]);
        return {up[u][0], res};
    }

    void reroot(int new_root) noexcept {
        if (new_root < 0 || new_root >= n || new_root == current_root) return;
        current_root = new_root;
        dfs_lca(current_root, current_root, 0, 0);
    }
};

} // namespace ae::mind
