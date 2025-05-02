class EstadoPuzzle:
    def __init__(self, tablero, estado_objetivo, padre=None, movimiento=""):
        self.tablero = tablero
        self.estado_objetivo = estado_objetivo  # Estado objetivo ahora es un parámetro
        self.padre = padre
        self.movimiento = movimiento
        
    def es_objetivo(self):
        # Compara el tablero actual con el estado objetivo
        return self.tablero == self.estado_objetivo

    def distancia_manhattan(self, estado):
        distancia = 0
        for i in range(3):
            for j in range(3):
                valor = estado[i][j]
                if valor != 0:  # Ignoramos el espacio vacío
                    x_objetivo = (valor - 1) // 3
                    y_objetivo = (valor - 1) % 3
                    distancia += abs(x_objetivo - i) + abs(y_objetivo - j)
        return distancia

    def obtener_posicion_vacia(self):
        for i in range(3):
            for j in range(3):
                if self.tablero[i][j] == 0:
                    return i, j
        return None

    def obtener_movimientos_posibles(self):
        movimientos = []
        x, y = self.obtener_posicion_vacia()
        direcciones = [('ARRIBA', -1, 0), ('ABAJO', 1, 0), ('IZQUIERDA', 0, -1), ('DERECHA', 0, 1)]
        
        for direccion, dx, dy in direcciones:
            nuevo_x, nuevo_y = x + dx, y + dy
            if 0 <= nuevo_x < 3 and 0 <= nuevo_y < 3:
                movimientos.append((direccion, nuevo_x, nuevo_y))
        return movimientos

    def resolver(self):
        estados_por_visitar = [(self.distancia_manhattan(self.tablero), 0, self)]  # (f_score, pasos, estado)
        visitados = set()
        contador_estados_visitados = 0  # Contador de estados visitados

        while estados_por_visitar:
            # Seleccionar el nodo con el menor f_score
            menor_f_score = min(estados_por_visitar, key=lambda x: x[0])
            estados_por_visitar.remove(menor_f_score)
            _, pasos, estado_actual = menor_f_score

            if estado_actual.es_objetivo():
                print(f"Estados visitados: {contador_estados_visitados}")
                return obtener_camino_solucion(estado_actual)

            tablero_actual = tuple(map(tuple, estado_actual.tablero))
            if tablero_actual in visitados:
                continue

            visitados.add(tablero_actual)
            contador_estados_visitados += 1  # Incrementar el contador

            for movimiento, nuevo_x, nuevo_y in estado_actual.obtener_movimientos_posibles():
                nuevo_tablero = [fila[:] for fila in estado_actual.tablero]
                x, y = estado_actual.obtener_posicion_vacia()
                nuevo_tablero[x][y], nuevo_tablero[nuevo_x][nuevo_y] = nuevo_tablero[nuevo_x][nuevo_y], nuevo_tablero[x][y]

                nuevo_estado = EstadoPuzzle(nuevo_tablero, self.estado_objetivo, estado_actual, movimiento)
                nuevo_tablero_tupla = tuple(map(tuple, nuevo_tablero))

                if nuevo_tablero_tupla not in visitados:
                    f_score = pasos + 1 + nuevo_estado.distancia_manhattan(nuevo_tablero)
                    estados_por_visitar.append((f_score, pasos + 1, nuevo_estado))
                    
        return None

def obtener_camino_solucion(estado):
    camino = []
    actual = estado
    while actual.padre:
        camino.append(actual.movimiento)
        actual = actual.padre
    camino.reverse()
    return camino

def imprimir_tablero(tablero):
    for fila in tablero:
        print(fila)
    print()

if __name__ == "__main__":
    # Estado inicial de prueba
    estado_inicial = [
        [1, 2, 0],
        [4, 6, 3],
        [7, 5, 8]
    ]
    
    # Estado objetivo definido como parámetro
    estado_objetivo = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ]
    
    print("Estado inicial:")
    imprimir_tablero(estado_inicial)
    
    # Crear instancia del puzzle con estado inicial y objetivo
    puzzle = EstadoPuzzle(estado_inicial, estado_objetivo)
    solucion = puzzle.resolver()
    
    if solucion:
        print("¡Solución encontrada!")
        print("Número de movimientos:", len(solucion))
        print("Movimientos:", solucion)
        print("Estado Final:")
        imprimir_tablero(estado_objetivo)
    else:
        print("¡No se encontró solución!")
