import copy  # Importa el módulo copy para realizar copias profundas de estructuras de datos.
from collections import deque  # Importa deque para usarlo como una cola FIFO.

# Clase Nodo representa un estado del puzzle y su relación con otros estados.
class Nodo:
    def __init__(self, estado, padre=None, movimiento=None, profundidad=0):
        self.estado = estado  # Estado actual del puzzle.
        self.padre = padre  # Nodo padre que generó este estado.
        self.movimiento = movimiento  # Movimiento que llevó a este estado.
        self.profundidad = profundidad  # Profundidad del nodo en el árbol de búsqueda.

    # Genera el camino de movimientos desde el estado inicial hasta este nodo.
    def generar_camino(self):
        camino = []  # Lista para almacenar los movimientos.
        nodo_actual = self  # Nodo actual para recorrer hacia atrás.
        while nodo_actual.padre is not None:  # Mientras no se llegue al nodo raíz.
            camino.append(nodo_actual.movimiento)  # Agrega el movimiento al camino.
            nodo_actual = nodo_actual.padre  # Retrocede al nodo padre.
        camino.reverse()  # Invierte el camino para que esté en orden correcto.
        return camino  # Devuelve el camino de movimientos.

# Clase PuzzleDeslizante implementa la lógica del puzzle y su resolución.
class PuzzleDeslizante:
    def __init__(self, estado_inicial, estado_objetivo):
        self.estado_inicial = estado_inicial  # Estado inicial del puzzle.
        self.estado_objetivo = estado_objetivo  # Estado objetivo del puzzle.
        # Diccionario que define los movimientos posibles y sus efectos en las coordenadas.
        self.movimientos = {
            "up": (-1, 0),  # Movimiento hacia arriba.
            "down": (1, 0),  # Movimiento hacia abajo.
            "left": (0, -1),  # Movimiento hacia la izquierda.
            "right": (0, 1)  # Movimiento hacia la derecha.
        }

    # Encuentra la posición del espacio vacío (representado por 0) en el estado actual.
    def encontrar_vacio(self, estado):
        for i, fila in enumerate(estado):  # Recorre las filas del estado.
            for j, valor in enumerate(fila):  # Recorre los valores de cada fila.
                if valor == 0:  # Si encuentra el espacio vacío.
                    return i, j  # Devuelve las coordenadas del espacio vacío.

    # Realiza un movimiento en el estado actual y devuelve el nuevo estado.
    def mover(self, estado, direccion):
        fila_vacia, col_vacia = self.encontrar_vacio(estado)  # Encuentra la posición del espacio vacío.
        delta_fila, delta_col = self.movimientos[direccion]  # Obtiene el cambio en las coordenadas según la dirección.
        nueva_fila, nueva_col = fila_vacia + delta_fila, col_vacia + delta_col  # Calcula las nuevas coordenadas.

        # Verifica si las nuevas coordenadas están dentro de los límites del puzzle.
        if 0 <= nueva_fila < len(estado) and 0 <= nueva_col < len(estado[0]):
            nuevo_estado = copy.deepcopy(estado)  # Crea una copia del estado actual.
            # Intercambia el espacio vacío con el valor en las nuevas coordenadas.
            nuevo_estado[fila_vacia][col_vacia], nuevo_estado[nueva_fila][nueva_col] = \
                nuevo_estado[nueva_fila][nueva_col], nuevo_estado[fila_vacia][col_vacia]
            return nuevo_estado  # Devuelve el nuevo estado.
        return None  # Devuelve None si el movimiento no es válido.

    # Resuelve el puzzle utilizando búsqueda en anchura (BFS).
    def resolver(self):
        frontera = set()  # Conjunto para almacenar los estados visitados.
        cola = deque([Nodo(self.estado_inicial)])  # Cola FIFO inicializada con el nodo raíz.
        contador_estados_visitados = 0  # Contador de estados visitados.

        while cola:  # Mientras haya nodos en la cola.
            nodo_actual = cola.popleft()  # Extrae el nodo actual de la cola.
            contador_estados_visitados += 1  # Incrementa el contador de estados visitados.

            if nodo_actual.estado == self.estado_objetivo:  # Si el estado actual es el objetivo.
                print(f"Estados visitados: {contador_estados_visitados}")  # Imprime el número de estados visitados.
                return nodo_actual.generar_camino()  # Devuelve el camino de movimientos.

            # Agrega el estado actual a los estados visitados.
            frontera.add(tuple(map(tuple, nodo_actual.estado)))

            # Genera nuevos estados a partir de los movimientos posibles.
            for movimiento in self.movimientos:
                nuevo_estado = self.mover(nodo_actual.estado, movimiento)  # Aplica el movimiento.
                # Si el nuevo estado es válido y no ha sido visitado.
                if nuevo_estado and tuple(map(tuple, nuevo_estado)) not in frontera:
                    # Crea un nuevo nodo con el nuevo estado.
                    nuevo_nodo = Nodo(nuevo_estado, nodo_actual, movimiento, nodo_actual.profundidad + 1)
                    cola.append(nuevo_nodo)  # Agrega el nuevo nodo a la cola.

        print(f"Estados visitados: {contador_estados_visitados}")  # Imprime el número de estados visitados si no hay solución.
        return None  # Devuelve None si no se encuentra solución.

# Ejemplo de uso del puzzle deslizante.
if __name__ == "__main__":
    # Define el estado inicial del puzzle.
    estado_inicial = [
        [1, 2, 0],
        [4, 6, 3],
        [7, 5, 8]
    ]
    # Define el estado objetivo del puzzle.
    estado_objetivo = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]

    # Crea una instancia del puzzle con el estado inicial y objetivo.
    puzzle = PuzzleDeslizante(estado_inicial, estado_objetivo)
    # Resuelve el puzzle y obtiene la solución.
    solucion = puzzle.resolver()

    # Función para imprimir una matriz en formato legible.
    def imprimir_matriz(estado):
        for fila in estado:  # Recorre las filas de la matriz.
            print(fila)  # Imprime cada fila.
        print()  # Imprime una línea en blanco para separar matrices.

    # Si se encuentra una solución.
    if solucion:
        print("Estado inicial:")  # Imprime el estado inicial.
        imprimir_matriz(estado_inicial)  # Muestra el estado inicial.
        print("Solución encontrada:", solucion)  # Imprime los movimientos de la solución.
        print()
        print("Estado objetivo:")  # Imprime el estado objetivo.
        imprimir_matriz(estado_objetivo)  # Muestra el estado objetivo.
        print("Cantidad de movimientos:", len(solucion))  # Imprime la cantidad de movimientos.
    else:
        print("No se encontró solución.")  # Mensaje si no hay solución.