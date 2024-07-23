#include <iostream>
#include <vector>
#include <sstream>
#include <stack>

using namespace std;

// Función para contar los caminos desde src hasta dest usando DFS iterativa
int countPaths(const vector<vector<int>>& graph, int src, int dest) {
    int n = static_cast<int>(graph.size());
    vector<bool> visited(n, false);
    stack<pair<int, int>> s; // (current_node, path_count)
    s.emplace(src, 0); // Usar emplace en lugar de push
    int paths = 0;

    while (!s.empty()) {
        auto [node, path_count] = s.top();
        s.pop();

        if (node == dest) {
            paths++;
            continue;
        }

        visited[node] = true;
        for (int i = 0; i < n; ++i) {
            if (graph[node][i] && !visited[i]) {
                s.emplace(i, path_count + 1); // Usar emplace en lugar de push
            }
        }
        visited[node] = false;
    }
    return paths;
}

int main() {
    string input = R"(
2

4
0 1 0 1
0 0 1 0
0 0 0 1
0 0 0 0
0 3

7
0 1 0 1 0 0 0
0 0 1 0 0 1 0
0 0 0 0 0 1 0
0 0 1 0 1 0 0
0 0 0 0 0 1 0
0 0 0 0 0 0 1
0 0 0 0 0 0 0
1 4
)";

    istringstream iss(input);
    int t;
    iss >> t;
    while (t--) {
        int n;
        iss >> n;
        vector<vector<int>> graph(n, vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                iss >> graph[i][j];
            }
        }
        int src, dest;
        iss >> src >> dest;
        cout << countPaths(graph, src, dest) << endl;
    }

    return 0;
}
