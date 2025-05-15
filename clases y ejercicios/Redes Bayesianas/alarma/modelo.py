from pomegranate import Node, DiscreteDistribution, ConditionalProbabilityTable, BayesianNetwork

# pip install pomegranate
# Si no funciona hay q instalar Microsoft Visual C++ 14 o mayor

Robo = Node(DiscreteDistribution({
    "robo": 0.001,
    "no_robo": 0.999
}), name="Robo")

Temblor = Node(DiscreteDistribution({
    "tiembla": 0.002,
    "no_tiembla": 0.998
}), name="Temblor")

Alarma = Node(ConditionalProbabilityTable([
    ["robo", "tiembla", "suena", 0.95],
    ["robo", "no_tiembla", "suena", 0.94],
    ["no_robo", "tiembla", "suena", 0.29],
    ["no_robo", "no_tiembla", "suena", 0.001],
    ["robo", "tiembla", "no_suena", 0.05],
    ["robo", "no_tiembla", "no_suena", 0.06],
    ["no_robo", "tiembla", "no_suena", 0.71],
    ["no_robo", "no_tiembla", "no_suena", 0.999]
], [Robo.distribution, Temblor.distribution]), name="Alarma")

Jorge = Node(ConditionalProbabilityTable([
    ["suena", "llama", 0.9],
    ["suena", "no_llama", 0.1],
    ["no_suena", "llama", 0.05],
    ["no_suena", "no_llama", 0.95]
], [Alarma.distribution]), name="Jorge")

Maria = Node(ConditionalProbabilityTable([
    ["suena", "llama", 0.7],
    ["suena", "no_llama", 0.3],
    ["no_suena", "llama", 0.01],
    ["no_suena", "no_llama", 0.99]
], [Alarma.distribution]), name="Maria")


# Creamos una Red Bayesiana y añadimos estados
modelo = BayesianNetwork()
modelo.add_states(Robo, Temblor, Alarma, Jorge, Maria)

# Añadimos bordes que conecten nodos

modelo.add_edge(Robo, Alarma)
modelo.add_edge(Temblor, Alarma)
modelo.add_edge(Alarma, Jorge)
modelo.add_edge(Alarma, Maria)

#Modelo Final
modelo.bake()
print("Modelo creado correctamente")