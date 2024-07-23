#include <iostream>
#include <unordered_set>
#include <sstream>

using namespace std;

// Función para verificar la propiedad de ventana
string tienePropiedadVentana(const string& secuencia) {
    size_t n = secuencia.size();
    for (size_t k = 1; k <= n; ++k) {
        unordered_set<string> patrones;
        for (size_t i = 0; i <= n - k; ++i) {
            patrones.insert(secuencia.substr(i, k));
        }
        if (patrones.size() > k + 1) {
            // Encontrar la posición del primer símbolo infractor
            for (size_t i = 0; i <= n - k; ++i) {
                patrones.clear();
                for (size_t j = 0; j < i; ++j) {
                    patrones.insert(secuencia.substr(j, k));
                }
                patrones.insert(secuencia.substr(i, k));
                if (patrones.size() > k + 1) {
                    return "NO:" + to_string(i + 1);
                }
            }
        }
    }
    return "SI";
}

int main() {
    string input = R"(
ababcababa
0010100100
0010101001
)";

    istringstream iss(input);
    string secuencia;
    while (getline(iss, secuencia)) {
        if (!secuencia.empty()) {
            cout << tienePropiedadVentana(secuencia) << endl;
        }
    }

    return 0;
}