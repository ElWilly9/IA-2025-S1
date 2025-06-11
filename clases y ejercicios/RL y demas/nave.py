import pygame
import numpy as np
import tensorflow as tf
from collections import deque
import random

# Game constants
WIDTH, HEIGHT = 400, 600
PLAYER_SIZE = 40
ENEMY_SIZE = 30
LASER_SIZE = 10
PLAYER_SPEED = 5
ENEMY_SPEED = 2
LASER_SPEED = 10
FPS = 60

# DQN Hyperparameters
REPLAY_MEMORY_SIZE = 100000  # Reduced for lower memory usage
MINIBATCH_SIZE = 16  # Reduced for faster training
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_FRAMES = 1000000
LEARNING_RATE = 0.00025
FRAME_SKIP = 4
TRAINING_FRAMES = 10000000
MAX_EPISODES = 1000  # Optional episode limit

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

class Player:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2 - PLAYER_SIZE // 2, HEIGHT - 60, PLAYER_SIZE, PLAYER_SIZE)
        self.lasers = []

    def move(self, dx):
        self.rect.x += dx
        self.rect.clamp_ip(screen.get_rect())

    def shoot(self):
        laser = pygame.Rect(self.rect.centerx - LASER_SIZE // 2, self.rect.top - LASER_SIZE, LASER_SIZE, LASER_SIZE)
        self.lasers.append(laser)

    def draw(self):
        pygame.draw.rect(screen, (0, 255, 0), self.rect)
        for laser in self.lasers:
            pygame.draw.rect(screen, (255, 255, 0), laser)

class Enemy:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, ENEMY_SIZE, ENEMY_SIZE)

    def move(self):
        self.rect.y += ENEMY_SPEED

    def draw(self):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)

class Game:
    def __init__(self):
        self.player = Player()
        self.enemies = []
        self.score = 0
        self.frame_count = 0
        self.spawn_timer = 0
        self.running = True

    def step(self, action):
        self.frame_count += 1
        reward = 0.1
        terminal = False

        if action == 0:
            self.player.move(-PLAYER_SPEED)
        elif action == 1:
            self.player.move(PLAYER_SPEED)
        elif action == 2:
            self.player.shoot()

        for laser in self.player.lasers[:]:
            laser.y -= LASER_SPEED
            if laser.bottom < 0:
                self.player.lasers.remove(laser)

        self.spawn_timer += 1
        if self.spawn_timer > 60:
            x = random.randint(0, WIDTH - ENEMY_SIZE)
            self.enemies.append(Enemy(x, 0))
            self.spawn_timer = 0

        for enemy in self.enemies[:]:
            enemy.move()
            if enemy.rect.top > HEIGHT:
                self.enemies.remove(enemy)
                reward -= 0.5
            elif enemy.rect.colliderect(self.player.rect):
                terminal = True
                reward -= 10
                self.running = False

        for laser in self.player.lasers[:]:
            for enemy in self.enemies[:]:
                if laser.colliderect(enemy.rect):
                    self.player.lasers.remove(laser)
                    self.enemies.remove(enemy)
                    reward += 1
                    self.score += 1
                    break

        return reward, terminal

    def render(self):
        screen.fill((0, 0, 0))
        self.player.draw()
        for enemy in self.enemies:
            enemy.draw()
        pygame.display.flip()

    def get_frame(self):
        frame = pygame.surfarray.array3d(screen)
        frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        frame = frame[::4, ::4]
        frame = frame[:100, :100]
        frame_surface = pygame.surfarray.make_surface(np.repeat(frame[:, :, np.newaxis], 3, axis=2))
        frame_surface = pygame.transform.scale(frame_surface, (84, 84))
        frame = pygame.surfarray.array3d(frame_surface)[:, :, 0]
        frame = frame / 255.0
        return frame

class DQN:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.frame_count = 0
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(84, 84, 4)),
            tf.keras.layers.Conv2D(32, (4, 4), strides=2, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE), loss='mse')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward, next_state, terminal):
        self.memory.append((state, action, reward, next_state, terminal))

    def choose_action(self, state):
        self.frame_count += 1
        if self.frame_count < EPSILON_DECAY_FRAMES:
            self.epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * self.frame_count / EPSILON_DECAY_FRAMES
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.memory) < MINIBATCH_SIZE:
            return
        minibatch = random.sample(self.memory, MINIBATCH_SIZE)
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        terminals = np.array([t[4] for t in minibatch])

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        for i in range(MINIBATCH_SIZE):
            if terminals[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + GAMMA * np.max(target_next[i])
        self.model.fit(states, targets, epochs=1, verbose=0)

def main_sync():
    game = Game()
    dqn = DQN(action_size=4)
    state_stack = deque(maxlen=4)
    for _ in range(4):
        state_stack.append(np.zeros((84, 84)))
    episode = 0
    while game.frame_count < TRAINING_FRAMES and episode < MAX_EPISODES:
        episode += 1
        game.__init__()
        state_stack = deque(maxlen=4)
        for _ in range(4):
            state_stack.append(game.get_frame())
        state = np.stack(state_stack, axis=-1)
        while game.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.event.pump()
            action = dqn.choose_action(state)
            total_reward = 0
            for _ in range(FRAME_SKIP):
                reward, terminal = game.step(action)
                total_reward += reward
                if game.frame_count % FRAME_SKIP == 0:
                    game.render()
                if terminal:
                    break
            next_frame = game.get_frame()
            state_stack.append(next_frame)
            next_state = np.stack(state_stack, axis=-1)
            total_reward = np.clip(total_reward, -1, 1)
            dqn.store_transition(state, action, total_reward, next_state, terminal)
            if game.frame_count % 4 == 0:  # Train every 4 steps
                dqn.train()
            state = next_state
            if terminal:
                break
            clock.tick(FPS)
        if game.frame_count % 10000 == 0:
            dqn.update_target_model()
        print(f"Episode {episode}, Score: {game.score}, Epsilon: {dqn.epsilon:.3f}")
    pygame.quit()

if __name__ == "__main__":
    main_sync()