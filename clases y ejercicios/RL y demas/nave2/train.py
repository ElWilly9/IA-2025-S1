from beam_rider_env import BeamRiderEnv
from dqn_agent import DQNAgent
from utils import initialize_stack, update_stack, save_model, plot_training
import numpy as np
import torch
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = BeamRiderEnv()
    agent = DQNAgent(n_actions=3)
    num_episodes = 501

    # Listas para almacenar las métricas
    episodios = []
    recompensas = []
    epsilons = []

    for ep in range(num_episodes):
        obs = env.reset()
        stack = initialize_stack(obs)
        total_reward = 0
        done = False

        while not done:
            state = np.stack(stack, axis=0)
            action = agent.act(state)
            next_obs, reward, done, _ = env.step(action)
            next_state_stack = update_stack(stack, next_obs)

            agent.remember(state, action, reward, next_state_stack, done)
            agent.replay()
            total_reward += reward

        # Guardar métricas
        episodios.append(ep)
        recompensas.append(total_reward)
        epsilons.append(agent.epsilon)

        if ep % 10 == 0:
            agent.update_target_network()
            print(f"Episode {ep} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

        if ep % 100 == 0 and ep > 0:
            save_model(agent, f"nave2/models/dqn_model_ep{ep}.pth")

    # Visualizar las métricas al final del entrenamiento
    plot_training(episodios, recompensas, epsilons)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
