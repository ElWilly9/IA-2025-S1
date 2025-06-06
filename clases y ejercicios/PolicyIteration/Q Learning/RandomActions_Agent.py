import numpy as np
import pandas as pd
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)

#creación del ambiente virtual con la paquetería gymnasium
env=gym.make("FrozenLake-v1",
             desc=["SFFF", "FHFF", "FFFH", "FFFG"],
             is_slippery=False,
             render_mode="human")

#Índices de las acciones
action_names = {0:'Izquierda', 1: 'Abajo', 2: 'Derecha', 3: 'Arriba'}

#Número de acciones
action_space_size = env.action_space.n
#Número de estados(cuadros)
state_space_size = env.observation_space.n

print('Ambiente:\n{}\n Núm. estados: {} \n Núm. acciones: {}'.format(env.get_wrapper_attr('desc'), state_space_size, action_space_size))

episodios = 10

for episodio in range(1, episodios+1):

    #Estado inicial
    estado = env.reset()
    done = False                    #done = False implica que el agente no ha llegado a la meta
    score = 0

    actions = []
    while not done:

        #visualizar el ambiente
        env.render()                #quitar comentario para visualizar

        #Muestrea una acción aleatoria
        action = env.action_space.sample()
        actions.append(action_names[action])

        #Actualización de las variables en el ambiente por la acción tomada
        _, reward, done, _, _ = env.step(action)
        score += reward

    print('Ensayo:{}\n\tAcciones:{}; Recompensa: {}'.format(episodio, actions, score))

