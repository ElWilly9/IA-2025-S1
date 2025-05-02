import copy

class PuzzleState:
    def __init__(self, board, parent=None, action=None, depth=0):
        self.board = board
        self.parent = parent
        self.action = action
        self.depth = depth
        self.heuristic = self.calculate_heuristic()
        self.key = self.calculate_key()

    def calculate_key(self):
        # Genera una clave única para cada estado del tablero.
        return tuple(map(tuple, self.board))

    def is_goal(self):
        goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        return self.board == goal_state

    def calculate_heuristic(self):
        h = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != 0:
                    goal_row = (self.board[i][j] - 1) // 3
                    goal_col = (self.board[i][j] - 1) % 3
                    h += abs(i - goal_row) + abs(j - goal_col)
        return h

    def generate_successors(self):
        successors = []
        zero_row, zero_col = self.find_zero()
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dr, dc in actions:
            new_row, new_col = zero_row + dr, zero_col + dc

            if 0 <= new_row < 3 and 0 <= new_col < 3:
                new_board = copy.deepcopy(self.board)
                new_board[zero_row][zero_col], new_board[new_row][new_col] = new_board[new_row][new_col], 0
                successors.append(PuzzleState(new_board, self, (dr, dc), self.depth + 1))
        return successors

    def find_zero(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j
                

                
    def solve(self):
        initial_state = PuzzleState(self.board)
        open_dict = ColaPrioridad()  # Crear una cola de prioridad
        open_dict.agregar(initial_state)  # Agregar el estado inicial a la cola de prioridad
        closed_set = set()

        while True:
            current_state = open_dict.eliminar()  # Eliminar el estado con la menor prioridad de la cola
            if current_state.is_goal():
                print("Solución encontrada:")
                return print_solution_steps(current_state)
                

            closed_set.add(current_state.key)
            successors = current_state.generate_successors()

            for successor in successors:
                if successor.key not in closed_set and successor.key not in open_dict.lista:
                    open_dict.agregar(successor)
                    
class ColaPrioridad:
    def __init__(self):
        self.lista = {}

    def agregar(self, state):
        self.lista[state.key] = state
        
    def eliminar(self):
        if len(self.lista) == 0:
            raise Exception("vacía")
        else:
            current_state_key = min(self.lista, key=lambda k: self.lista[k].heuristic + self.lista[k].depth)
            current_state = self.lista.pop(current_state_key)
            return current_state

def print_board(board):
    for row in board:
        print(" ".join(map(str, row)))
    print("\n")

# ...

def print_solution_steps(solution):
    if solution is None:
        print("No se encontró solución.")
    else:
        current = solution
        steps = []

        while current.parent:
            steps.append(current)
            current = current.parent

        steps.reverse()

        for step in steps:
            print(f"Paso {step.depth}:")
            print_board(step.board)
            print("Heurística:", step.heuristic)
            print("Profundidad:", step.depth)
            print("Acción:", step.action)
            print("-" * 20)
            
            
            
if __name__ == "__main__":
    initial_board = [[8, 6, 4],
                     [2, 1, 7], 
                     [3, 5, 0]]
    
    initial_state = PuzzleState(initial_board)
    initial_state.solve()
