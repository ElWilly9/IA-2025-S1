import sys

class Nodo():
    def __init__(self, estado, padre, accion):
        self.estado = estado
        self.padre = padre
        self.accion = accion

class Frontera():
    def __init__(self):
        self.frontera =[]
    
    def empty(self):
        return (len(self.frontera) == 0)

    def add(self, nodo):
        self.frontera.append(nodo)
    
    def eliminar(self):
        # LIFO o FIFO
        pass

    def contiene_estado(self, estado):
        return any(nodo.estado == estado for nodo in self.frontera)

class Pila(Frontera):
    def eliminar(self):
        # Termina la busqueda si la frontera esta vacia
        if self.empty():        
            raise Exception("Frontera vacia")
        else:
            # Guardamos el ultimo item en la lista 
            # (el cual es el nodo recientemente añadido)
            nodo = self.frontera[-1]
            # Guardamos todos los items excepto el 
            # ultimo (eliminamos)
            self.frontera = self.frontera[:-1]
            return nodo
    
class Cola(Frontera):
    def eliminar(self):
        # Termina la busqueda si la frontera esta vacia
        if self.empty():        
            raise Exception("Frontera vacia")
        else:
            # Guardamos el primer item en la lista 
            # (el cual es el nodo añadido de primero)
            nodo = self.frontera[0]
            # Guardamos todos los items excepto el 
            # primero (eliminamos)
            self.frontera = self.frontera[1:]
            return nodo
    
class Laberinto():
    def __init__(self, filename):  
        # Leemos el archivo y establecemos las dimensiones del laberinto
        with open(filename) as file:
            contenido = file.read()
        
        # Verificamos punto inicial y objetivo
        if contenido.count("A") != 1:
            raise Exception("El laberinto debe tener solo un punto de inicio")
        if contenido.count("B") != 1:
            raise Exception("El laberinto debe ter tener exactamente un objetivo")

        # Determinemos el alto y ancho del laberinto
        contenido = contenido.splitlines()
        self.altura = len(contenido)
        self.ancho = max(len(line) for line in contenido)
        print(contenido)
        
        # Control de los muros
        self.muros = []
        for i in range(self.altura):
            filas = []
            for j in range(self.ancho):
                try:
                    if contenido[i][j] == "A":
                        self.inicio = (i,j)             # Estado inicial
                        filas.append(False)
                    elif contenido[i][j] == "B":
                        self.objetivo = (i,j)           # Estado objetivo
                        filas.append(False)
                    elif contenido[i][j] == " ":
                        filas.append(False)
                    else:
                        filas.append(True)
                except IndexError:
                    filas.append(False)
            self.muros.append(filas)

        self.solucion = None

    def print(self):
        sol = self.solucion[1] if self.solucion is not None else None
        print()
        for i, fila in enumerate(self.muros):
            for j, col in enumerate(fila):
                if col:
                    print("█", end="")
                elif (i,j) == self.inicio:
                    print("A", end="")
                elif (i,j) == self.objetivo:
                    print("B", end="")
                elif sol is not None and (i,j) in sol:
                    print("*", end="")
                else:
                    print(" ", end="")
            print()
        print()
                    
    def vecinos(self, estado):
        fila, col = estado
        candidatos = [
            ("up", (fila -1, col)),
            ("down", (fila +1, col)),
            ("left", (fila, col -1)),
            ("right", (fila, col +1))
        ]

        resultados = []
        for accion, (f,c) in candidatos:
            if 0<=f < self.altura and 0<=c < self.ancho and not self.muros[f][c]:
                resultados.append((accion,(f,c)))
        return resultados


    def solve(self):
        # Encuentra una solución al laberinto si existe #
        
        # Lleva el registro del número de estados explorados
        self.num_explorados = 0

        # Inicializamos la frontera para empezar en la posicion inicial
        start = Nodo(estado=self.inicio, padre=None, accion=None)
        frontera = Pila()
        frontera.add(start)
        
        # Inicializamos en conjunto explorado vacio
        self.explorado = set()

        # Manetenmos el bucle hasta que encontremos la solución
        while True:

            # Si nada queda en la frontera, entonces no hya mas camino
            if frontera.empty():
                raise Exception("No hay Solución")
            
            # Escogemos un nodo de la frontera
            nodo = frontera.eliminar()
            self.num_explorados +=1

            # Si el nodo es el objetivo, entonces tenemos una solución
            if nodo.estado == self.objetivo:
                acciones = []
                cel = []
                # Rastreamos los nodos padre hasta la solución (objetivo hasta estado inicial)
                while nodo.padre is not None:
                    acciones.append(nodo.accion)
                    cel.append(nodo.estado)
                    nodo = nodo.padre
                acciones.reverse()
                cel.reverse()
                self.solucion = (acciones, cel)
                return
            
            # Marcamos el nodo como exploado
            self.explorado.add(nodo.estado)

            # Agregamos vecinos a la frontera
            for accion, estado in self.vecinos(nodo.estado):
                if not frontera.contiene_estado(estado) and estado not in self.explorado:
                    hijo = Nodo(estado = estado, padre = nodo, accion=accion)
                    frontera.add(hijo)

            
# Main 
if len(sys.argv) != 2:
    sys.exit("Utiliza: python laberinto.py laberinto.txt")

l = Laberinto(sys.argv[1])
print("Laberiinto:")
l.print()
print("Resolviendo...")
l.solve()
print("Estados explorados:", l.num_explorados)
print("Solucion")
l.print()