import numpy as np
import pandas as pd

def micro_blackjack_value_iteration(max_iterations=4):
    """
    Resuelve el problema de Micro-Blackjack usando Value Iteration
    Estados: {0, 2, 3, 4, 5}
    Acciones: Draw, Stop
    Cartas: {2, 3, 4} con probabilidad 1/3 cada una
    """
    
    # Estados posibles
    states = [0, 2, 3, 4, 5]
    actions = ['Draw', 'Stop']
    cards = [2, 3, 4]
    card_prob = 1/3
    
    # Inicializar valores
    V = {s: 0 for s in states}
    
    # Para almacenar la historia de iteraciones
    iterations_history = []
    
    def get_reward(state, action):
        """Función de recompensa"""
        if action == 'Stop':
            return state
        else:  # Draw
            return 0  # No hay recompensa inmediata por sacar carta
    
    def get_next_states(state, action):
        """Obtiene los posibles estados siguientes y sus probabilidades"""
        if action == 'Stop':
            return [(state, 1.0)]  # Te quedas en el mismo estado
        
        next_states = []
        for card in cards:
            next_state = state + card
            if next_state > 5:
                # Te pasas - vas al estado "perdedor" (valor 0)
                next_states.append((0, card_prob))
            else:
                next_states.append((next_state, card_prob))
        return next_states
    
    # Guardar valores iniciales
    current_values = V.copy()
    iterations_history.append(current_values.copy())
    
    print("=== MICRO-BLACKJACK VALUE ITERATION ===\n")
    print("Estados: suma de cartas {0, 2, 3, 4, 5}")
    print("Acciones: Draw (sacar carta), Stop (detenerse)")
    print("Cartas disponibles: {2, 3, 4} con probabilidad 1/3 cada una")
    print("Objetivo: maximizar utilidad (puntaje sin pasarse de 5)\n")
    
    # Value Iteration
    for iteration in range(1, max_iterations + 1):
        print(f"--- ITERACIÓN {iteration} ---")
        
        V_new = {}
        
        for state in states:
            if state == 5:
                # Estado terminal - solo puedes detenerte
                V_new[state] = 5
                print(f"Estado {state}: Terminal, V = 5")
                continue
            
            action_values = {}
            
            # Evaluar acción STOP
            stop_value = get_reward(state, 'Stop')
            action_values['Stop'] = stop_value
            
            # Evaluar acción DRAW
            draw_value = get_reward(state, 'Draw')
            next_states = get_next_states(state, 'Draw')
            
            expected_future_value = 0
            for next_state, prob in next_states:
                expected_future_value += prob * V[next_state]
            
            draw_value += expected_future_value
            action_values['Draw'] = draw_value
            
            # Tomar la mejor acción
            best_value = max(action_values.values())
            best_action = max(action_values, key=action_values.get)
            
            V_new[state] = best_value
            
            print(f"Estado {state}:")
            print(f"  Stop: {action_values['Stop']:.3f}")
            print(f"  Draw: {action_values['Draw']:.3f}")
            print(f"  Mejor: {best_action} (V = {best_value:.3f})")
        
        V = V_new
        iterations_history.append(V.copy())
        print()
    
    # Crear tabla de resultados
    print("=== TABLA DE ITERACIONES ===")
    
    # Crear DataFrame
    df_data = []
    for i, values in enumerate(iterations_history):
        row = [i] + [values[state] for state in states]
        df_data.append(row)
    
    df = pd.DataFrame(df_data, columns=['Iteración'] + [f'V{s}' for s in states])
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Política óptima
    print("\n=== POLÍTICA ÓPTIMA ===")
    for state in states:
        if state == 5:
            print(f"Estado {state}: STOP (terminal)")
            continue
            
        # Calcular valores de acciones para la política final
        stop_value = state
        
        draw_value = 0
        for card in cards:
            next_state = state + card
            if next_state > 5:
                draw_value += card_prob * 0  # Pierdes
            else:
                draw_value += card_prob * V[next_state]
        
        if stop_value >= draw_value:
            action = "STOP"
            value = stop_value
        else:
            action = "DRAW"
            value = draw_value
            
        print(f"Estado {state}: {action} (V = {value:.3f})")
    
    return df, V

# Ejecutar el algoritmo
df_results, final_values = micro_blackjack_value_iteration(max_iterations=4)

print(f"\n=== ANÁLISIS DE RESULTADOS ===")
print("La política óptima sugiere:")
print("- Estado 0: DRAW (no tienes nada, vale la pena arriesgar)")
print("- Estado 2: Depende de los cálculos, pero probablemente DRAW")
print("- Estado 3: Depende de los cálculos, podría ser STOP o DRAW")
print("- Estado 4: Probablemente STOP (riesgo alto de pasarse)")
print("- Estado 5: STOP (óptimo, no puedes mejorar)")