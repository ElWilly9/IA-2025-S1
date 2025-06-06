import numpy as np
import pandas as pd
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(12345)

action_names = ['Izquierda', 'Abajo', 'Derecha', 'Arriba']

# creación del ambiente del agente que aprenderá con Q-learning
env=gym.make("FrozenLake-v1",desc=["SFFF", "FHFF", "FFFH", "FFFG"], is_slippery=False, render_mode="ansi")

env.action_space.n

env.observation_space.n

Q_table = np.zeros((env.observation_space.n,env.action_space.n ))

#para visualizar la Q-table como un data frame
df=pd.DataFrame(data=Q_table, columns=['Izquierda', 'Abajo', 'Derecha', 'Arriba'])
df.index = [i+1 for i in range(len(df))]
df.columns.name = 'Estado'
print(df)

#Tasa de aprendizaje
def learning_rate(n : int , min_rate=0.01 ) -> float  :
    """Tasa de decaimiento del aprendizaje por cada episodio"""

    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

#Tasa de exploración
def exploration_rate(n : int , min_rate=0.01 ) -> float  :
    """Tasa de decaimiento de la exploración por cada episodio"""

    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))

#Politica epsilon-greedy
def policy_e_greedy(current_state : tuple):
    """Regla de decisión basada en una política epsilon-greedy, dado en el estado que se está"""

    # explorar (acción aleatoria)
    if np.random.random() < exploration_rate(e):
        action = env.action_space.sample()

    # explotar
    else:
        action = np.argmax(Q_table[current_state])

    return action

# Actualización del Q-value
def new_Q_value( reward : float , action, current_state, new_state : tuple , discount_factor=1 ) -> float:
    """
    Actualización del Q-value de un par estado-acción con el método de diferencias temporales:

       reward = recompensa obtenida en el estado actual.
       action = acción realizada
       current_state = estado actual
       new_state = estado siguiente
       discount_factor = el peso que se le da al estado siguiente (entre más cercano a 1, más importa).

    """
    act_val = Q_table[current_state, action]                             #se obtiene el valor del estado actual
    fut_opt_val = np.max(Q_table[new_state])                             #se obtiene el valor del estado siguiente
    learned_value = reward + discount_factor * fut_opt_val-act_val       # r(s, a) + γmax Q' (s',a') - Q(s, a)
    new_val = (1-lr)*act_val + lr*(learned_value)

    return new_val

#Etapa de entrenamiento/aprendizaje

num_episodio = []         #lista donde se irán guardando el número de episodio
puntajes_ep = []          #lista donde se irán guardando los puntajes por episodio

n_episodes = 500         #se entrenará al agente con 1000 episodios

for e in range(1, n_episodes+1):

    #Estado inicial
    current_state = env.reset()[0]
    done = False
    num_episodio.append(e)
    score = 0

    actions = []
    while done==False:

        #Se selecciona la acción con la política epsilon greedy
        action = policy_e_greedy(current_state)
        actions.append(action_names[action])

        #Actualización de las variables en el ambiente
        #a partir de la acción tomada
        obs, reward, done, _, _ = env.step(action)

        new_state = obs
        score += reward
        lr = learning_rate(e)                         #se actualiza la tasa de aprendizaje

        #Se actualiza el valor Q
        Q_table[current_state, action] = new_Q_value(reward, action, current_state, new_state)

        #Se transita al nuevo estado
        current_state = new_state

    puntajes_ep.append(score)

    #impresión de resultados cada 100 episodios
    if e % 100 == 0:
        puntaje = puntajes_ep[e-1]

        print('Episode {}\n\tPuntaje:{}; Acciones:{}'.format(e, score, actions))

# Visualización de los resultados
plt.plot(puntajes_ep,'.')
plt.title('Scores en cada episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa')
plt.show()

#visualizacion después de entrenar
from seaborn import heatmap

datos = pd.DataFrame(data=Q_table, columns=['Izquierda', 'Abajo', 'Derecha', 'Arriba'])
heatmap(datos, cmap='Blues', annot=True, square=False)
plt.ylabel('Estados')
plt.show()

#prueba de aprendizaje
#Se crea el ambiente virtual
env=gym.make("FrozenLake-v1",desc=["SFFF", "FHFF", "FFFH", "FFFG"], is_slippery=False, render_mode="human")

#Se evalúa el agente en 5 episodios

n_episodes_t = 5

for e in range(1, n_episodes_t+1):

    current_state = env.reset()[0]
    done = False
    score = 0

    actions = []
    while done==False:

        # política de acción
        action = np.argmax(Q_table[current_state]) #la accion del agente es siempre explotar
        actions.append(action_names[action])

        #se obtienen los resultados del ambiente por la acción elegida
        obs, reward, done, _,_= env.step(action)
        score += reward

        #se transita al nuevo estado
        current_state = obs

    print('Episodio: {}\n\tAcciones: {}; Puntaje: {}'.format(e, actions, score))

env.close()

