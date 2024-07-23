#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std;

// Función para calcular el total de pasto que comerá la primera vaca
int maxGrass(const vector<int>& blocks) {
    int n = static_cast<int>(blocks.size());  // Conversión explícita a int
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int i = 0; i < n; ++i) {
        dp[i][i] = blocks[i];
    }

    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i <= n - len; ++i) {
            int j = i + len - 1;
            dp[i][j] = max(blocks[i] + min((i + 2 <= j) ? dp[i + 2][j] : 0, (i + 1 <= j - 1) ? dp[i + 1][j - 1] : 0),
                           blocks[j] + min((i <= j - 2) ? dp[i][j - 2] : 0, (i + 1 <= j - 1) ? dp[i + 1][j - 1] : 0));
        }
    }
    return dp[0][n - 1];
}

int main() {
    string input = R"(
3
4
12 15 13 11
8
2 2 1 5 3 8 7 3
6
2 5 3 1 3 1
)";
    istringstream iss(input);
    int t;
    iss >> t;
    while (t--) {
        int n;
        iss >> n;
        vector<int> blocks(n);
        for (int i = 0; i < n; ++i) {
            iss >> blocks[i];
        }
        cout << maxGrass(blocks) << endl;
    }
    return 0;
}