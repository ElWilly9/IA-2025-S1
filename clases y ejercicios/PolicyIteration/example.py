import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#creación del entorno
env=gym.make("FrozenLake-v1",desc=["SFFF", "FHFF", "FFHF", "FFFG"], is_slippery=True, render_mode="human")
#Acciones
action_names = {0:'Izquierda', 1:'Abajo', 2:'Derecha',3:'Arriba'}

print('Número de estados: ', env.observation_space)
# acciones: izquierda = 0, abajo = 1, derecha = 2 y arriba = 3
print('Número de acciones: ', env.action_space)

env.unwrapped.P[0][1]

def policy_evaluation(env, policy_vec, val_fun_vec, conv_tolerance, discount=0.9):
    """Evaluación de la política"""
    while True:
        delta = 0
        for state in env.unwrapped.P:
            prev_val = val_fun_vec[state]
            #Cálculo del retorno esperando
            expected_return = sum([prob*(reward + discount*val_fun_vec[next_state]) 
                                      for prob,next_state,reward,is_terminal_state 
                                      in env.unwrapped.P[state][policy_vec[state]]])
            #Actualización de los valores
            val_fun_vec[state] = expected_return
            #Actualización de delta
            delta = np.max([delta, np.abs(prev_val - val_fun_vec[state])])

        if(delta < conv_tolerance):
            return policy_vec, val_fun_vec, delta
        
def policy_improvement(policy_vec, val_fun_vec, policy_stable_vec, discount=0.9):
    """Mejoramiento de la política"""
    for state in env.unwrapped.P:
        old_action = policy_vec[state]
        #Cálculo de los valores de acción
        action_values = np.zeros(env.action_space.n)
        for action_p in env.unwrapped.P[state]:
            action_values[action_p] = sum([prob*(reward + discount*val_fun_vec[next_state]) 
                                           for prob, next_state, reward, is_terminal_state 
                                           in env.unwrapped.P[state][action_p]])
        #Acción de mayor valor
        policy_vec[state] = np.argmax(action_values)
        
        if old_action == policy_vec[state]:
            policy_stable_vec[state] = True
            
    return policy_vec, policy_stable_vec

def policy_iteration(env, discount = 0.9, conv_tolerance=1e-4, max_iterations=100):
    """Algoritmo de iteración de política"""
    #Vector de estabilidad
    policy_stable_vec = np.zeros(env.observation_space.n, dtype=bool)
    #Función de valor
    val_fun_vec = np.zeros(env.observation_space.n)
    #Política
    #policy_vec = np.ones(env.observation_space.n, dtype=int)
    policy_vec = np.zeros(env.observation_space.n, dtype=int)
    for iterations in range(max_iterations):
        #Evaluación de política
        policy_vec, val_fun_vec, delta = policy_evaluation(env, policy_vec, val_fun_vec, 
                                                                       conv_tolerance, discount=discount)
        #Mejoramiento de la política
        policy_vec, policy_stable_vec = policy_improvement(policy_vec, val_fun_vec, policy_stable_vec, 
                                                           discount=discount)
        
        if np.all(policy_stable_vec == True):
            return policy_vec, val_fun_vec

policy_vec, val_fun_vec = policy_iteration(env, discount=0.5)

def values_print(valueFunction,reshapeDim=4):
    ax = sns.heatmap(valueFunction.reshape(reshapeDim,reshapeDim),annot=True, square=True,cbar=False, 
                     cmap='Blues',xticklabels=False, yticklabels=False)
    plt.title('Función de valor por cada estado')
    plt.show()

def actions_print(policy_vec,reshapeDim=4):
    ax = sns.heatmap(policy_vec.reshape(reshapeDim,reshapeDim),annot=np.array([action_names[a] for a in policy_vec]).reshape(reshapeDim,reshapeDim), 
                     fmt='',cbar=False, cmap='Oranges',xticklabels=False, yticklabels=False)
    plt.title('Política en cada estado')
    plt.show()

values_print(val_fun_vec)
actions_print(policy_vec)

#Prueba de la política
env.reset()
n_episodes_t = 2
for e in range(1, n_episodes_t+1):
    current_state = env.reset()[0]
    done = False
    score = 0
    actions = []
    while done == False:
        #Acción ambiciosa
        action = policy_vec[current_state]
        actions.append(action_names[action])
        #Se obtienen los resultados del ambiente por la acción elegida
        state, reward, done, _,_= env.step(action)
        score += reward
        #Se transita al nuevo estado
        current_state = state

    print('Episodio: {}\n\tAcciones: {};\n\tPuntaje: {}'.format(e, actions, score))

env.close()