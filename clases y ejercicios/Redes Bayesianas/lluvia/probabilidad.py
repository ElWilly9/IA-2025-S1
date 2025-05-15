from modelo import modelo

observaciones = [["ninguna", "si", "a tiempo", "atendida"],
                 ["ninguna", "si", "a tiempo", "perdida"],
                 ["fuerte", "no", "retrasada", "atendida"],
                 ]
# Calcular la probabildiad para 3 diferentes observaciones 

for observacion in observaciones:
   probability = modelo.probability([observacion])
   print(f"Probabilidad de {observacion}: {probability:.4f}")