"""
Jugador de Triqui
"""

import math

X = "X"
O = "O"
VACIO = None


def estado_inicial():
    """
    Retorna el estado inicial del tablero.
    """
    return [[VACIO, VACIO, VACIO],
            [VACIO, VACIO, VACIO],
            [VACIO, VACIO, VACIO]]

#turnos
def jugador(tablero):
    """
    Retorna el jugador que tiene el siguiente turno en el tablero.
    """
    # Contar X y O en el tablero
    x_count = sum(row.count(X) for row in tablero)
    o_count = sum(row.count(O) for row in tablero)
    
    # Si hay más X que O, le toca a O, y viceversa
    return O if x_count > o_count else X

#movimientos posibles
def acciones(tablero):
    """
    Retorna el conjunto de todas las acciones posibles (i, j) disponibles en el tablero.
    """
    acciones_posibles = set()
    for i in range(3):
        for j in range(3):
            if tablero[i][j] == VACIO:
                acciones_posibles.add((i, j))
    return acciones_posibles


def resultado(tablero, accion):
    """
    Retorna el tablero que resulta de realizar el movimiento (i, j) en el tablero.
    """
    # Verificar que la acción es válida
    if accion not in acciones(tablero):
        raise Exception("Acción inválida")
    
    # Crear una copia profunda del tablero
    nuevo_tablero = [row[:] for row in tablero]
    
    # Realizar el movimiento
    i, j = accion
    nuevo_tablero[i][j] = jugador(tablero)
    
    return nuevo_tablero


def ganador(tablero):
    """
    Retorna el ganador del juego, si lo hay.
    """
    # Verificar filas
    for i in range(3):
        if tablero[i][0] == tablero[i][1] == tablero[i][2] != VACIO:
            return tablero[i][0]
    
    # Verificar columnas
    for j in range(3):
        if tablero[0][j] == tablero[1][j] == tablero[2][j] != VACIO:
            return tablero[0][j]
    
    # Verificar diagonales
    if tablero[0][0] == tablero[1][1] == tablero[2][2] != VACIO:
        return tablero[0][0]
    if tablero[0][2] == tablero[1][1] == tablero[2][0] != VACIO:
        return tablero[0][2]
    
    return None


def final(tablero):
    """
    Retorna True si el juego ha terminado, False en caso contrario.
    """
    # Si hay un ganador, el juego ha terminado
    if ganador(tablero) is not None:
        return True
    
    # Si no hay espacios vacíos, el juego ha terminado
    if not acciones(tablero):
        return True
    
    return False


def utilidad(tablero):
    """
    Retorna 1 si X ha ganado, -1 si O ha ganado, 0 en otro caso.
    """
    ganador_juego = ganador(tablero)
    if ganador_juego == X:
        return 1
    elif ganador_juego == O:
        return -1
    else:
        return 0


def minimax(tablero):
    """
    Retorna la acción óptima para el jugador actual en el tablero.
    """
    if final(tablero):
        return None
    
    if jugador(tablero) == X:
        valor, accion = valor_max_alfa_beta(tablero, float('-inf'), float('inf'))
    else:
        valor, accion = valor_min_alfa_beta(tablero, float('-inf'), float('inf'))
    
    return accion


def valor_max_alfa_beta(tablero, alfa, beta):
    """
    Retorna el valor máximo posible para el jugador X usando poda alfa-beta.
    """
    if final(tablero):
        return utilidad(tablero), None
    
    valor = float('-inf')
    mejor_accion = None
    
    for accion in acciones(tablero):
        nuevo_valor, _ = valor_min_alfa_beta(resultado(tablero, accion), alfa, beta)
        if nuevo_valor > valor:
            valor = nuevo_valor
            mejor_accion = accion
        
        # Poda alfa-beta
        alfa = max(alfa, valor)
        if alfa >= beta:
            break
    
    return valor, mejor_accion


def valor_min_alfa_beta(tablero, alfa, beta):
    """
    Retorna el valor mínimo posible para el jugador O usando poda alfa-beta.
    """
    if final(tablero):
        return utilidad(tablero), None
    
    valor = float('inf')
    mejor_accion = None
    
    for accion in acciones(tablero):
        nuevo_valor, _ = valor_max_alfa_beta(resultado(tablero, accion), alfa, beta)
        if nuevo_valor < valor:
            valor = nuevo_valor
            mejor_accion = accion
        
        # Poda alfa-beta
        beta = min(beta, valor)
        if alfa >= beta:
            break
    
    return valor, mejor_accion
