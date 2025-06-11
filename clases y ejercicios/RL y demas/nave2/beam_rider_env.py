import gym
from gym import spaces
import numpy as np
import pygame
import random
import cv2

SCREEN_WIDTH, SCREEN_HEIGHT = 84, 84

class BeamRiderEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(3)  # 0: izquierda, 1: quedarse, 2: derecha
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.player_x = SCREEN_WIDTH // 2
        self.enemies = []
        self.score = 0
        self.done = False
        self.frame_count = 0
        return self._get_obs()

    def step(self, action):
        if action == 0: self.player_x -= 5
        elif action == 2: self.player_x += 5
        self.player_x = np.clip(self.player_x, 0, SCREEN_WIDTH - 5)

        if random.random() < 0.1:
            self.enemies.append([random.randint(0, SCREEN_WIDTH - 5), 0])

        self.enemies = [[x, y+3] for x, y in self.enemies if y < SCREEN_HEIGHT]

        reward = 0
        for ex, ey in self.enemies:
            if abs(ex - self.player_x) < 5 and ey > SCREEN_HEIGHT - 10:
                self.done = True
                reward = -1
                break

        self.score += 1
        reward += 0.1
        self.frame_count += 1
        if self.frame_count > 1000:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        obs = np.zeros((84, 84), dtype=np.uint8)
        for x, y in self.enemies:
            obs[y:y+2, x:x+2] = 255
        obs[SCREEN_HEIGHT - 5:SCREEN_HEIGHT, self.player_x:self.player_x+5] = 255
        return obs[..., np.newaxis]

    def render(self):
        frame = self._get_obs()
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Beam Rider", frame)
        cv2.waitKey(1)  # Espera 1 ms

