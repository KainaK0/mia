#include <iostream>
#include <vector>
#include <sstream>

using namespace std;

// Función para procesar cada campo del buscaminas
void processField(int n, int m, const vector<string>& field, int fieldNum) {
    vector<vector<int>> result(n, vector<int>(m, 0));

    // Recorrer cada celda del campo
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (field[i][j] == '*') {
                // Incrementar el contador en las celdas adyacentes
                for (int di = -1; di <= 1; ++di) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        int ni = i + di, nj = j + dj; // Coordenadas de la celda adyacente
                        // Verificar si la celda adyacente está dentro del campo y no es una mina
                        if (ni >= 0 && ni < n && nj >= 0 && nj < m && field[ni][nj] != '*') {
                            result[ni][nj]++; // Incrementar el contador de minas adyacentes
                        }
                    }
                }
            }
        }
    }

    // Imprimir el campo procesado
    cout << "Field #" << fieldNum << ":" << endl;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (field[i][j] == '*') {
                cout << '*'; // Imprimir la mina
            } else {
                cout << result[i][j]; // Imprimir el número de minas adyacentes
            }
        }
        cout << endl; // Nueva línea para la siguiente fila del campo
    }
}

int main() {
    string input = R"(
4 4
*...
....
.*..
....

3 5
**...
.....
.*...

0 0
)";

    istringstream iss(input);
    int n, m, fieldNumber = 1;
    while (true) {
        iss >> n >> m;
        if (n == 0 && m == 0) {
            break; // Condición de salida
        }

        vector<string> field(n);
        for (int i = 0; i < n; ++i) {
            iss >> field[i];
        }

        if (fieldNumber > 1) cout << endl; // Línea en blanco entre campos
        processField(n, m, field, fieldNumber++); // Procesar el campo de minas
    }

    return 0;
}
