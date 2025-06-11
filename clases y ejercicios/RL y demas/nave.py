import numpy as np
import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

class BeamRiderEnvironment:
    """Entorno simplificado del juego Beam Rider"""
    
    def __init__(self, width=84, height=84):
        self.width = width
        self.height = height
        self.action_space = 4  # 0: no hacer nada, 1: izquierda, 2: derecha, 3: disparar
        
        # Inicializar pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width * 6, height * 6))
        pygame.display.set_caption("Beam Rider DQN")
        self.clock = pygame.time.Clock()
        
        self.reset()
    
    def reset(self):
        """Reiniciar el juego"""
        self.player_x = self.width // 2
        self.player_y = self.height - 10
        self.bullets = []
        self.enemies = []
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.steps = 0
        self.enemy_spawn_counter = 0
        
        # Generar algunos enemigos iniciales
        for _ in range(3):
            self.spawn_enemy()
        
        return self.get_state()
    
    def spawn_enemy(self):
        """Generar un nuevo enemigo"""
        x = random.randint(5, self.width - 5)
        y = random.randint(5, self.height // 2)
        speed = random.uniform(0.5, 1.5)
        self.enemies.append({'x': x, 'y': y, 'speed': speed})
    
    def step(self, action):
        """Ejecutar un paso del juego"""
        self.steps += 1
        reward = 0
        
        # Procesar acción del jugador
        if action == 1 and self.player_x > 2:  # Izquierda
            self.player_x -= 2
        elif action == 2 and self.player_x < self.width - 2:  # Derecha
            self.player_x += 2
        elif action == 3:  # Disparar
            self.bullets.append({'x': self.player_x, 'y': self.player_y - 5})
        
        # Actualizar balas
        self.bullets = [b for b in self.bullets if b['y'] > 0]
        for bullet in self.bullets:
            bullet['y'] -= 3
        
        # Actualizar enemigos
        for enemy in self.enemies:
            enemy['y'] += enemy['speed']
        
        # Generar nuevos enemigos
        self.enemy_spawn_counter += 1
        if self.enemy_spawn_counter > 60:  # Cada 60 frames
            self.spawn_enemy()
            self.enemy_spawn_counter = 0
        
        # Eliminar enemigos que salieron de la pantalla
        self.enemies = [e for e in self.enemies if e['y'] < self.height + 5]
        
        # Detectar colisiones bala-enemigo
        for bullet in self.bullets[:]:
            for enemy in self.enemies[:]:
                if (abs(bullet['x'] - enemy['x']) < 3 and 
                    abs(bullet['y'] - enemy['y']) < 3):
                    self.bullets.remove(bullet)
                    self.enemies.remove(enemy)
                    self.score += 10
                    reward += 10
                    break
        
        # Detectar colisiones jugador-enemigo
        for enemy in self.enemies:
            if (abs(self.player_x - enemy['x']) < 4 and
                abs(self.player_y - enemy['y']) < 4):
                self.lives -= 1
                reward -= 50
                self.enemies.remove(enemy)
                if self.lives <= 0:
                    self.game_over = True
                    reward -= 100
                break
        
        # Recompensa por supervivencia
        reward += 0.1
        
        # Penalización por inactividad
        if action == 0:
            reward -= 0.05
        
        return self.get_state(), reward, self.game_over
    
    def get_state(self):
        """Obtener el estado actual del juego como imagen"""
        state = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Dibujar jugador
        if 0 <= self.player_x < self.width and 0 <= self.player_y < self.height:
            state[self.player_y, self.player_x] = 1.0
        
        # Dibujar balas
        for bullet in self.bullets:
            x, y = int(bullet['x']), int(bullet['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                state[y, x] = 0.7
        
        # Dibujar enemigos
        for enemy in self.enemies:
            x, y = int(enemy['x']), int(enemy['y'])
            if 0 <= x < self.width and 0 <= y < self.height:
                state[y, x] = -1.0
        
        return state
    
    def render(self):
        """Renderizar el juego"""
        self.screen.fill((0, 0, 0))
        
        # Dibujar jugador (verde)
        pygame.draw.rect(self.screen, (0, 255, 0), 
                        (self.player_x * 6, self.player_y * 6, 6, 6))
        
        # Dibujar balas (amarillo)
        for bullet in self.bullets:
            pygame.draw.rect(self.screen, (255, 255, 0), 
                           (bullet['x'] * 6, bullet['y'] * 6, 3, 6))
        
        # Dibujar enemigos (rojo)
        for enemy in self.enemies:
            pygame.draw.rect(self.screen, (255, 0, 0), 
                           (enemy['x'] * 6, enemy['y'] * 6, 6, 6))
        
        # Mostrar información
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(lives_text, (10, 50))
        
        pygame.display.flip()
        self.clock.tick(60)


class DQN(nn.Module):
    """Red neuronal Deep Q-Network"""
    
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        # Capas convolucionales (basadas en el paper)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        
        # Calcular el tamaño después de las convoluciones
        conv_out_size = self._get_conv_out(input_shape)
        
        # Capas completamente conectadas
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, n_actions)
        
    def _get_conv_out(self, shape):
        """Calcular el tamaño de salida de las capas convolucionales"""
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    """Agente DQN con Experience Replay"""
    
    def __init__(self, state_shape, n_actions, lr=0.00025, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.000001,
                 memory_size=100000, batch_size=32):
        
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Redes neuronales
        self.q_network = DQN(state_shape, n_actions).to(device)
        self.target_network = DQN(state_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Actualizar target network
        self.update_target_network()
        
        # Historial de estados para el preprocessing
        self.state_history = deque(maxlen=4)
        
    def update_target_network(self):
        """Actualizar la red objetivo"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def preprocess_state(self, state):
        """Preprocesar estado - apilar 4 frames"""
        # Añadir el estado actual al historial
        self.state_history.append(state)
        
        # Si no tenemos suficientes frames, rellenar con el estado actual
        while len(self.state_history) < 4:
            self.state_history.append(state)
        
        # Apilar los 4 frames
        stacked_state = np.stack(self.state_history, axis=0)
        return stacked_state
    
    def remember(self, state, action, reward, next_state, done):
        """Almacenar experiencia en memoria"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Seleccionar acción usando epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Entrenar la red con experience replay"""
        if len(self.memory) < self.batch_size:
            return
        
        # Muestrear batch aleatorio
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        
        # Q-values actuales
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Q-values siguientes
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Calcular pérdida
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decaer epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
    
    def save_model(self, filepath):
        """Guardar modelo"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        """Cargar modelo"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def train_dqn():
    """Función principal de entrenamiento"""
    print("Iniciando entrenamiento DQN para Beam Rider")
    
    # Parámetros
    episodes = 1000
    target_update_freq = 1000
    save_freq = 100
    
    # Crear entorno y agente
    env = BeamRiderEnvironment()
    state_shape = (4, 84, 84)  # 4 frames apilados
    agent = DQNAgent(state_shape, env.action_space)
    
    # Métricas
    scores = []
    avg_scores = []
    epsilons = []
    
    # Crear directorio para guardar modelos
    os.makedirs('models', exist_ok=True)
    
    for episode in range(episodes):
        state = env.reset()
        processed_state = agent.preprocess_state(state)
        total_reward = 0
        steps = 0
        
        while True:
            # Renderizar cada 10 episodios
            if episode % 10 == 0:
                env.render()
            
            # Seleccionar acción
            action = agent.act(processed_state)
            
            # Ejecutar acción
            next_state, reward, done = env.step(action)
            next_processed_state = agent.preprocess_state(next_state)
            
            # Almacenar experiencia
            agent.remember(processed_state, action, reward, next_processed_state, done)
            
            # Entrenar
            agent.replay()
            
            processed_state = next_processed_state
            total_reward += reward
            steps += 1
            
            # Manejar eventos de pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            if done or steps > 1000:
                break
        
        # Actualizar target network
        if episode % target_update_freq == 0:
            agent.update_target_network()
            print(f"Target network actualizada en episodio {episode}")
        
        # Guardar modelo
        if episode % save_freq == 0 and episode > 0:
            model_path = f'models/dqn_beam_rider_episode_{episode}.pth'
            agent.save_model(model_path)
            print(f"Modelo guardado: {model_path}")
        
        # Registrar métricas
        scores.append(total_reward)
        avg_scores.append(np.mean(scores[-100:]))
        epsilons.append(agent.epsilon)
        
        # Mostrar progreso
        if episode % 10 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episodio {episode}, Score: {total_reward:.2f}, "
                  f"Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Memoria: {len(agent.memory)}")
    
    # Guardar modelo final
    agent.save_model('models/dqn_beam_rider_final.pth')
    
    # Mostrar gráficas
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.title('Puntuación por Episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Puntuación')
    
    plt.subplot(2, 2, 2)
    plt.plot(avg_scores)
    plt.title('Puntuación Promedio (últimos 100 episodios)')
    plt.xlabel('Episodio')
    plt.ylabel('Puntuación Promedio')
    
    plt.subplot(2, 2, 3)
    plt.plot(epsilons)
    plt.title('Epsilon (Exploración)')
    plt.xlabel('Episodio')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    pygame.quit()


def play_game():
    """Jugar con un modelo entrenado"""
    print("Modo de juego - Cargando modelo entrenado...")
    
    # Crear entorno y agente
    env = BeamRiderEnvironment()
    state_shape = (4, 84, 84)
    agent = DQNAgent(state_shape, env.action_space)
    
    # Cargar modelo entrenado
    try:
        agent.load_model('models/dqn_beam_rider_final.pth')
        agent.epsilon = 0.05  # Poca exploración para evaluación
        print("Modelo cargado exitosamente")
    except:
        print("No se pudo cargar el modelo. Usando modelo aleatorio.")
    
    # Jugar varios episodios
    for episode in range(10):
        state = env.reset()
        processed_state = agent.preprocess_state(state)
        total_reward = 0
        steps = 0
        
        print(f"\nEpisodio {episode + 1}")
        
        while True:
            env.render()
            
            # Seleccionar acción
            action = agent.act(processed_state, training=False)
            
            # Ejecutar acción
            next_state, reward, done = env.step(action)
            next_processed_state = agent.preprocess_state(next_state)
            
            processed_state = next_processed_state
            total_reward += reward
            steps += 1
            
            # Manejar eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        done = True
            
            if done or steps > 2000:
                break
        
        print(f"Puntuación final: {total_reward:.2f}, Pasos: {steps}")
    
    pygame.quit()


if __name__ == "__main__":
    print("=== BEAM RIDER CON DEEP Q-LEARNING ===")
    print("1. Entrenar modelo")
    print("2. Jugar con modelo entrenado")
    
    choice = input("Selecciona una opción (1 o 2): ")
    
    if choice == "1":
        train_dqn()
    elif choice == "2":
        play_game()
    else:
        print("Opción no válida")