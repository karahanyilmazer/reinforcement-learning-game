import random
from collections import deque

import numpy as np
from sympy import print_rcode
import torch
from helper import plot
from model import HybridQNet,HybridQTrainer, Linear_QNet, QTrainer

from game import Direction, Point, SnakeGameAI

MAX_MEMORY = 1000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self, is_explore = True, is_pretrained = True):
        self.is_explore = is_explore
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
        self.model = HybridQNet(11, 256, 3).to(self.device)
        # self.is_pretrained = is_pretrained
        # if self.is_pretrained:
        #     weights = torch.load('./model/model_nobound_85.pth')
        #     self.model.load_state_dict(weights)
        self.trainer = HybridQTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # Danger right
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d)),
            # Danger left
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, frame, state, action, reward, next_frame, next_state, done):
        self.memory.append(
            (frame, state, action, reward, next_frame, next_state, done)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        frames, states, actions, rewards, next_frames, next_states, dones = zip(*mini_sample)
        self.trainer.train_step( frames, states, actions, rewards, next_frames, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, frame, state, action, reward, next_frame, next_state, done):
        self.trainer.train_step(frame, state, action, reward, next_frame, next_state, done)

    def get_action(self, state, frame):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 500 - self.n_games
        final_move = [0, 0, 0]
        if (random.randint(0, 200) < self.epsilon) and self.is_explore:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            frame0 = torch.tensor(frame, dtype=torch.float).to(self.device)

            prediction = self.model(frame0, state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1 # type: ignore

        return final_move

def add_dim(array):
    array = array[None, ...]
    return array

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(is_explore=True, is_pretrained=True)
    game = SnakeGameAI(w=160, h=160, is_bounds = False)
    select = input('Press Enter to start or type anything to select simulation mode \n')
    if select != '':
        explore = input('Exploration y/n \n')
        agent.is_explore = True if explore == 'y' else False
        bounds = input('Bounds y/n \n')
        game.is_bounds = True if bounds == 'y' else False

    while True:
        # get old state
        state_old = agent.get_state(game)
        frame_old = game.capture_frame().T

        # get move
        final_move = agent.get_action(add_dim(state_old), add_dim(frame_old))
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        frame_new = game.capture_frame().T
        # train short memory
        agent.train_short_memory(add_dim(frame_old), add_dim(state_old), final_move, reward, add_dim(frame_new), add_dim(state_new), done)

        # remember
        agent.remember(frame_old, state_old, final_move, reward, frame_new, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
