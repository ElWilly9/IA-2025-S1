from modelo import modelo

observaciones = [["robo", "no_tiembla", "suena", "llama", "no_llama"],
                ["no_robo", "no_tiembla", "suena", "no_llama", "llama"],
                ["no_robo", "tiembla", "no_suena", "llama", "no_llama"],
                ["no_robo", "tiembla", "suena", "no_llama", "llama"]
                 ]
# Calcular la probabildiad para 3 diferentes observaciones 

for observacion in observaciones:
   probability = modelo.probability([observacion])
   print(f"Probabilidad de {observacion}: {probability:.9f}")