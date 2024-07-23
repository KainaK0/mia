#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

// Función para encontrar el patrón de bits original
string findOriginalPattern(const vector<string>& fragments) {
    vector<string> candidates;

    for (const auto& fragment : fragments) {
        for (const auto& fragment2 : fragments) {
            if (fragment != fragment2) {
                // Intentar concatenar fragment y fragment2 en ambos órdenes
                string combined1 = fragment + fragment2;
                string combined2 = fragment2 + fragment;
                candidates.push_back(combined1);
                candidates.push_back(combined2);
            }
        }
    }

    // Filtrar los candidatos para encontrar el más corto que contiene todos los fragmentos
    string bestPattern;
    for (const auto& candidate : candidates) {
        bool valid = true;
        for (const auto& fragment : fragments) {
            if (candidate.find(fragment) == string::npos) {
                valid = false;
                break;
            }
        }
        if (valid && (bestPattern.empty() || candidate.size() < bestPattern.size())) {
            bestPattern = candidate;
        }
    }
    return bestPattern;
}

int main() {
    string input = R"(
1
011
0111
01110
111
0111
10111
)";

    istringstream iss(input);
    int t;
    iss >> t;
    iss.ignore(); // Ignorar la línea en blanco después del número de casos

    while (t--) {
        vector<string> fragments;
        string fragment;
        while (getline(iss, fragment)) {
            if (fragment.empty()) break;
            fragments.push_back(fragment);
        }
        cout << findOriginalPattern(fragments) << endl;
    }

    return 0;
}