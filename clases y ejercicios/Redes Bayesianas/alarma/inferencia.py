from modelo import modelo

# Calculemos las predicciones
predicciones = modelo.predict_proba({
    "Temblor": "no_tiembla"
})

# Visualizemos las predicciones para cada nodo
for nodo, prediccion in zip(modelo.states, predicciones):
    if isinstance(prediccion, str):
        print(f"{nodo.name}: {prediccion}")
    else:
        print(f"{nodo.name}")
        for valor, probabilidad in prediccion.parameters[0].items(): 
            print(f"       {valor}: {probabilidad:.2f}")