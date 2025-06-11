import numpy as np
import cv2
from collections import deque
import torch
import matplotlib.pyplot as plt

def preprocess_frame(frame, size=(84, 84)):
    """Convierte una imagen a escala de grises y la redimensiona."""
    if frame.shape[-1] == 3:  # imagen RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:  # imagen 1 canal pero 3D
        frame = frame.squeeze()
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def initialize_stack(initial_frame, stack_size=4):
    """Inicializa el stack de frames con el primer frame repetido."""
    processed = preprocess_frame(initial_frame)
    stack = deque([processed] * stack_size, maxlen=stack_size)
    return stack

def update_stack(stack, new_frame):
    """Actualiza el stack de frames con uno nuevo."""
    processed = preprocess_frame(new_frame)
    stack.append(processed)
    return stack  # devuelve el deque actualizado

def save_model(agent, filename="dqn_model.pth"):
    torch.save(agent.q_net.state_dict(), filename)

def load_model(agent, filename="dqn_model.pth"):
    agent.q_net.load_state_dict(torch.load(filename))
    agent.target_net.load_state_dict(agent.q_net.state_dict())

def plot_training(episodios, recompensas, epsilons):
    """
    Visualiza la evolución de los episodios, recompensas y valores de epsilon.
    Args:
        episodios (list o array): Lista de episodios.
        recompensas (list o array): Recompensa obtenida en cada episodio.
        epsilons (list o array): Valor de epsilon en cada episodio.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa', color=color)
    ax1.plot(episodios, recompensas, color=color, label='Recompensa')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Instancia un segundo eje que comparte el mismo eje x

    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)
    ax2.plot(episodios, epsilons, color=color, linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Evolución de Recompensa y Epsilon por Episodio')
    plt.show()
