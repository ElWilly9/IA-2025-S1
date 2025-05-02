
# Initial and goal states
estado_inicial = [
    [1, 2, 0],
    [4, 6, 3],
    [7, 5, 8]
]

estado_objetivo = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

# Function to print a board
def imprimir_tablero(tablero):
    for fila in tablero:
        print(fila)
    print()

## Algorithm 3: Breadth-First Search (BFS)

import
from collections import deque

class Nodo:
    def __init__(self, estado, padre=None, movimiento=None, profundidad=0):
        self.estado = estado
        self.padre = padre
        self.movimiento = movimiento
        self.profundidad = profundidad

    def generar_camino(self):
        camino = []
        nodo_actual = self
        while nodo_actual.padre is not None:
            camino.append(nodo_actual.movimiento)
            nodo_actual = nodo_actual.padre
        camino.reverse()
        return camino

class PuzzleDeslizante:
    def __init__(self, estado_inicial, estado_objetivo):
        self.estado_inicial = estado_inicial
        self.estado_objetivo = estado_objetivo
        self.movimientos = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }

    def encontrar_vacio(self, estado):
        for i, fila in enumerate(estado):
            for j, valor in enumerate(fila):
                if valor == 0:
                    return i, j

    def mover(self, estado, direccion):
        fila_vacia, col_vacia = self.encontrar_vacio(estado)
        delta_fila, delta_col = self.movimientos[direccion]
        nueva_fila, nueva_col = fila_vacia + delta_fila, col_vacia + delta_col

        if 0 <= nueva_fila < len(estado) and 0 <= nueva_col < len(estado[0]):
            nuevo_estado = copy.deepcopy(estado)
            nuevo_estado[fila_vacia][col_vacia], nuevo_estado[nueva_fila][nueva_col] = \
                nuevo_estado[nueva_fila][nueva_col], nuevo_estado[fila_vacia][col_vacia]
            return nuevo_estado
        return None

    def resolver(self):
        frontera = set()
        cola = deque([Nodo(self.estado_inicial)])
        contador_estados_visitados = 0

        while cola:
            nodo_actual = cola.popleft()
            contador_estados_visitados += 1

            if nodo_actual.estado == self.estado_objetivo:
                return nodo_actual.generar_camino(), contador_estados_visitados

            frontera.add(tuple(map(tuple, nodo_actual.estado)))

            for movimiento in self.movimientos:
                nuevo_estado = self.mover(nodo_actual.estado, movimiento)
                if nuevo_estado and tuple(map(tuple, nuevo_estado)) not in frontera:
                    nuevo_nodo = Nodo(nuevo_estado, nodo_actual, movimiento, nodo_actual.profundidad + 1)
                    cola.append(nuevo_nodo)

        return None, contador_estados_visitados

# Run BFS algorithm
print("Running Breadth-First Search (BFS)")
print("Estado inicial:")
imprimir_tablero(estado_inicial)
puzzle_bfs = PuzzleDeslizante(estado_inicial, estado_objetivo)
solucion_bfs, estados  estados_visitados_bfs = puzzle_bfs.resolver()

if solucion_bfs:
    print("Solución encontrada:", solucion_bfs)
    print("Cantidad de movimientos:", len(solucion_bfs))
    print("Estados visitados:", estados_visitados_bfs)
    print("Estado objetivo:")
    imprimir_tablero(estado_objetivo)
else:
    print("No se encontró solución.")
```

## Performance Comparison

This cell collects the performance metrics (states visited and number of steps) from each algorithm and creates a bar plot to compare them.

```python
import matplotlib.pyplot as plt
import numpy as np

# Collect performance data
algoritmos = ['A*', 'DFS', 'BFS']
estados_visitados = [
    estados_visitados_astar,
    estados_visitados_dfs,
    estados_visitados_bfs
]
pasos = [
    len(solucion_astar) if solucion_astar else 0,
    len(solucion_dfs) if solucion_dfs else 0,
    len(solucion_bfs) if solucion_bfs else 0
]

# Set up the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot states visited
x = np.arange(len(algoritmos))
ax1.bar(x, estados_visitados, color=['blue', 'orange', 'green'])
ax1.set_xlabel('Algoritmo')
ax1.set_ylabel('Estados Visitados')
ax1.set_title('Comparación de Estados Visitados')
ax1.set_xticks(x)
ax1.set_xticklabels(algoritmos)

# Plot number of steps
ax2.bar(x, pasos, color=['blue', 'orange', 'green'])
ax2.set_xlabel('Algoritmo')
ax2.set_ylabel('Número de Pasos')
ax2.set_title('Comparación de Número de Pasos')
ax2.set_xticks(x)
ax2.set_xticklabels(algoritmos)

plt.tight_layout()
plt.savefig('puzzle_comparison.png')
```

## Notes

- The A* algorithm uses the Manhattan distance heuristic to guide the search, making it more efficient in terms of states visited compared to uninformed search methods like BFS and DFS.
- DFS may not find the optimal solution due to its depth-limited nature and LIFO strategy, which can lead to exploring deep, non-optimal paths.
- BFS guarantees the optimal solution (shortest path) but may visit more states than A* due to its exhaustive exploration of all nodes at the current depth before moving deeper.
- The plot is saved as `puzzle_comparison.png` in the working directory.

To run this notebook:
1. Ensure you have Jupyter, Python, and the required libraries (`matplotlib`, `numpy`) installed.
2. Copy this code into a `.ipynb` file or create a new Jupyter notebook and paste the cells.
3. Execute each cell in order to see the output of each algorithm and the final comparison plot.