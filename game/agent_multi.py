import random
from collections import deque

import numpy as np
import torch
from helper import plot_multi

from game import Direction, Point, SnakeGameAI
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game, snake_num):
        if snake_num == 1:
            head = game.snake1[0]
            direction = game.direction1
        elif snake_num == 2:
            head = game.snake2[0]
            direction = game.direction2

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = direction == Direction.LEFT
        dir_r = direction == Direction.RIGHT
        dir_u = direction == Direction.UP
        dir_d = direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r, snake_num))
            or (dir_l and game.is_collision(point_l, snake_num))
            or (dir_u and game.is_collision(point_u, snake_num))
            or (dir_d and game.is_collision(point_d, snake_num)),
            # Danger right
            (dir_u and game.is_collision(point_r, snake_num))
            or (dir_d and game.is_collision(point_l, snake_num))
            or (dir_l and game.is_collision(point_u, snake_num))
            or (dir_r and game.is_collision(point_d, snake_num)),
            # Danger left
            (dir_d and game.is_collision(point_r, snake_num))
            or (dir_u and game.is_collision(point_l, snake_num))
            or (dir_r and game.is_collision(point_u, snake_num))
            or (dir_l and game.is_collision(point_d, snake_num)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores1 = []
    plot_scores2 = []
    plot_mean_scores1 = []
    plot_mean_scores2 = []
    total_score1 = 0
    total_score2 = 0
    record1 = 0
    record2 = 0
    agent1 = Agent()
    agent2 = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old1 = agent1.get_state(game, snake_num=1)
        state_old2 = agent2.get_state(game, snake_num=2)

        # get move
        final_move1 = agent1.get_action(state_old1)
        final_move2 = agent2.get_action(state_old2)

        # perform move and get new state
        reward1, done1, score1, reward2, done2, score2 = game.play_step(
            final_move1, final_move2
        )
        state_new1 = agent1.get_state(game, snake_num=1)
        state_new2 = agent2.get_state(game, snake_num=2)

        # train short memory
        agent1.train_short_memory(state_old1, final_move1, reward1, state_new1, done1)
        agent2.train_short_memory(state_old2, final_move2, reward2, state_new2, done2)

        # remember
        agent1.remember(state_old1, final_move1, reward1, state_new1, done1)
        agent2.remember(state_old2, final_move2, reward2, state_new2, done2)

        if done1 and done2:
            # train long memory, plot result
            game.reset()
            agent1.n_games += 1
            agent2.n_games += 1
            agent1.train_long_memory()
            agent2.train_long_memory()

            if score1 > record1:
                record1 = score1
                agent1.model.save(title='snake_1')
            if score2 > record2:
                record2 = score2
                agent2.model.save()

            print(
                'Game:',
                agent1.n_games,
                ' | Score:',
                score1,
                '-',
                score2,
                ' | Record:',
                record1,
                '-',
                record2,
            )

            plot_scores1.append(score1)
            plot_scores2.append(score2)
            total_score1 += score1
            total_score2 += score2
            mean_score1 = total_score1 / agent1.n_games
            mean_score2 = total_score2 / agent2.n_games
            plot_mean_scores1.append(mean_score1)
            plot_mean_scores2.append(mean_score2)
            plot_multi(plot_scores1, plot_mean_scores1, plot_scores2, plot_mean_scores2)


if __name__ == '__main__':
    train()
