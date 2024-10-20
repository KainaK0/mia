# Instalar la librería micrograd
# !pip install micrograd  # Install the micrograd library

# Importar los módulos necesarios
import random
from micrograd.engine import Value
from micrograd.nn import MLP

# Definir los datos de entrada (Xs) y las salidas deseadas (ys)
Xs = [
    [2.5, 3.5, -0.5],
    [4.0, -1.0, 0.5],
    [0.5, 1.5, 1.0],
    [3.0, 2.0, -1.5]
]
ys = [1.0, -1.0, -1.0, 1.0]  # Valores deseados

# Crear una red neuronal con 3 entradas, 2 capas ocultas de 4 neuronas y una salida de 1 neurona
random.seed(42)  # Para tener resultados reproducibles
n = MLP(3, [4, 4, 1])

# Definir la tasa de aprendizaje
learning_rate = 0.01

# Número de épocas (iteraciones de entrenamiento)
epochs = 200

# Función para imprimir los pesos actuales
def print_weights(n, etapa):
    print(f"\nPesos en la {etapa}:")
    for i, p in enumerate(n.parameters()):
        print(f"Peso {i+1}: {p.data:.4f}")

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Inicializar la pérdida
    loss = Value(0.0)

    # Forward pass y cálculo de la pérdida
    for x, y in zip(Xs, ys):
        y_pred = n(x)  # Predicción de la red
        loss += (y_pred - Value(y))**2  # Pérdida cuadrática

    # Mostrar los pesos en la penúltima y última iteración (forward propagation)
    if epoch in [epochs - 2, epochs - 1]:
        print(f'\nEpoch {epoch+1}, Forward pass')
        print(f'Pérdida: {loss.data / len(Xs)}')
        print_weights(n, f"forward propagation en la iteración {epoch+1}")

    # Backward pass (retropropagación)
    n.zero_grad()  # Inicializar los gradientes a cero antes de la retropropagación
    loss.backward()

    # Mostrar los gradientes (backward propagation)
    if epoch in [epochs - 2, epochs - 1]:
        print(f'\nEpoch {epoch+1}, Backward pass')
        print("Gradientes después de backward propagation:")
        for i, p in enumerate(n.parameters()):
            print(f"Gradiente del peso {i+1}: {p.grad:.4f}")

    # Actualización de los pesos
    for p in n.parameters():
        p.data -= learning_rate * p.grad  # Actualizar los parámetros (pesos) con gradiente descendente

    # Mostrar los pesos después de la actualización (backward propagation)
    if epoch in [epochs - 2, epochs - 1]:
        print_weights(n, f"backward propagation en la iteración {epoch+1}")

# Predicciones finales después del entrenamiento
print("\nPredicciones finales:")
for x, y in zip(Xs, ys):
    y_pred = n(x)
    print(f"Entrada: {x}, Predicción: {y_pred.data:.4f}, Objetivo: {y}")