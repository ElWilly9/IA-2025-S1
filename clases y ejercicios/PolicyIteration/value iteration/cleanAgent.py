import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#creación del entorno
env = gym.make("FrozenLake-v1",desc=["SF", "FG"], is_slippery=False, render_mode="human")
#Acciones
action_names = {0:'Arriba', 1:'Abajo', 2:'Derecha',3:'Izquierda'}

print('Número de estados: ', env.observation_space)
# acciones: izquierda = 0, abajo = 1, derecha = 2 y arriba = 3
print('Número de acciones: ', env.action_space)

env.unwrapped.P = {
    0: {  # Estado A
        0: [(1.0, 0, -1, False)],                     # arriba → se queda en A
        1: [(0.8, 2, -1, False), (0.2, 0, -1, False)], # abajo → C
        2: [(0.8, 1, -1, False), (0.2, 0, -1, False)], # derecha → B
        3: [(1.0, 0, -1, False)]                      # izquierda → se queda en A
    },
    1: {  # Estado B
        0: [(1.0, 1, -1, False)],                     
        1: [(0.8, 3, 10, True), (0.2, 1, -1, False)],  # abajo → D (terminal)
        2: [(1.0, 1, -1, False)],                     
        3: [(0.8, 0, -1, False), (0.2, 1, -1, False)]
    },
    2: {  # Estado C
        0: [(0.8, 0, -1, False), (0.2, 2, -1, False)],
        1: [(1.0, 2, -1, False)],                     
        2: [(0.8, 3, 10, True), (0.2, 2, -1, False)],  # derecha → D (terminal)
        3: [(1.0, 2, -1, False)]
    },
    3: {  # Estado D
        0: [(1.0, 3, 0, True)],
        1: [(1.0, 3, 0, True)],
        2: [(1.0, 3, 0, True)],
        3: [(1.0, 3, 0, True)]
    }
}


# La salida es una lista, donde cada entrada representa una de las acciones probables dada la acción elegida con las siguientes variables:
# (probabilidad de transición, siguiente estado, recompensa, ¿Es un estado terminal?)
env.unwrapped.P[0][1]

def policy_evaluation(policy, P, discount=0.9, tol=1e-4):
    V = np.zeros(len(P))
    while True:
        delta = 0
        for s in P:
            v = 0
            for prob, next_s, reward, done in P[s][policy[s]]:
                v += prob * (reward + discount * V[next_s])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    return V

def policy_improvement(V, P, discount=0.9):
    policy = np.zeros(len(P), dtype=int)
    for s in P:
        action_returns = []
        for a in env.action_space:
            q = 0
            for prob, next_s, reward, done in P[s][a]:
                q += prob * (reward + discount * V[next_s])
            action_returns.append(q)
        policy[s] = np.argmax(action_returns)
    return policy

def policy_iteration(P, discount=0.9, tol=1e-4):
    policy = np.ones(len(P), dtype=int)  # política aleatoria inicial
    while True:
        V = policy_evaluation(policy, P, discount, tol)
        new_policy = policy_improvement(V, P, discount)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, V

#ejecución del algoritmo de policy iteration
policy_vec, val_fun_vec = policy_iteration(env.unwrapped.P, discount=0.9, tol=1e-4)

def values_print(valueFunction,reshapeDim=2):
    ax = sns.heatmap(valueFunction.reshape(reshapeDim,reshapeDim),annot=True, square=True,cbar=False, 
                     cmap='Blues',xticklabels=False, yticklabels=False)
    plt.title('Función de valor por cada estado')
    plt.show()

def actions_print(policy_vec,reshapeDim=2):
    ax = sns.heatmap(policy_vec.reshape(reshapeDim,reshapeDim),annot=np.array([action_names[a] for a in policy_vec]).reshape(reshapeDim,reshapeDim), 
                     fmt='',cbar=False, cmap='Oranges',xticklabels=False, yticklabels=False)
    plt.title('Política en cada estado')
    plt.show()

values_print(val_fun_vec)
actions_print(policy_vec)

env.reset()
n_episodes_t = 20
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
