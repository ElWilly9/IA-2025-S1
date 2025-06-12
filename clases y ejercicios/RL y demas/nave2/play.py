import torch
import numpy as np
import cv2
from collections import deque
from beam_rider_env import BeamRiderEnv
from dqn_agent import DQNAgent
from utils import preprocess_frame, update_stack, initialize_stack, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    env = BeamRiderEnv()
    agent = DQNAgent(n_actions=3)
    load_model(agent, "nave2/models/dqn_model_ep500.pth")

    for episode in range(10):  # puedes cambiar el número de episodios a visualizar
        obs = env.reset()
        frame_stack = initialize_stack(obs)
        total_reward = 0
        done = False

        while not done:
            state = np.stack(frame_stack, axis=0)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = agent.q_net(state_tensor).argmax().item()

            obs, reward, done, _ = env.step(action)
            frame_stack.append(preprocess_frame(obs))
            total_reward += reward

            # Mostrar el entorno
            frame = np.stack(frame_stack, axis=0)[-1]  # última frame
            frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Beam Rider - Agent Playing", frame_resized)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                done = True
                break

        print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
