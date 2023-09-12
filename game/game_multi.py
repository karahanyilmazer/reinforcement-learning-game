import os
import random
from collections import namedtuple
from enum import Enum

import numpy as np
import pygame

pygame.init()
font = pygame.font.Font(os.path.join('game', 'arial.ttf'), 25)
# font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:
    def __init__(self, w=640, h=480, is_bounds = False):
        self.w = w
        self.h = h
        self.is_bounds = is_bounds
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.RIGHT

        self.head1 = Point(self.w / 2, self.h / 2)

        self.snake1 = [
            self.head1,
            Point(self.head1.x - BLOCK_SIZE, self.head1.y),
            Point(self.head1.x - (2 * BLOCK_SIZE), self.head1.y),
        ]
        self.head2 = Point(self.w / 4, self.h / 2)

        self.snake2 = [
            self.head2,
            Point(self.head2.x - BLOCK_SIZE, self.head2.y),
            Point(self.head2.x - (2 * BLOCK_SIZE), self.head2.y),
        ]

        self.score1 = 0
        self.score2 = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)

        if self.food in (self.snake1 or self.snake2):
            self._place_food()

    def play_step(self, action1, action2):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move1(action1)  # update the head
        self._move2(action2)  # update the head
        self.snake1.insert(0, self.head1)
        self.snake2.insert(0, self.head2)

        # 3. check if game over
        reward1, reward2 = 0, 0
        game_over1, game_over2 = False, False

        collision1 = self.is_collision(self.snake1[0], 1)
        collision2 = self.is_collision(self.snake2[0], 2)

        longest_len = max(len(self.snake1), len(self.snake2))

        # Game in endless loop
        if self.frame_iteration > 100 * longest_len:
            game_over1, game_over2 = True, True
            reward1, reward2 = -10, -10
            return reward1, game_over1, self.score1, reward2, game_over2, self.score2

        # Snake 1 crashed
        if collision1:
            game_over1, game_over2 = True, True
            reward1, reward2 = -10, 5
            return (
                reward1,
                game_over1,
                self.score1,
                reward2,
                game_over2,
                self.score2,
            )

        # Snake 2 crashed
        if collision2:
            game_over1, game_over2 = True, True
            reward1, reward2 = 5, -10
            return (
                reward1,
                game_over1,
                self.score1,
                reward2,
                game_over2,
                self.score2,
            )

        # 4. place new food or just move
        if self.head1 == self.food:
            self.score1 += 1
            reward1, reward2 = 10, -10
            self._place_food()
        else:
            self.snake1.pop()

        if self.head2 == self.food:
            self.score2 += 1
            reward1, reward2 = -10, 10
            self._place_food()
        else:
            self.snake2.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward1, game_over1, self.score1, reward2, game_over2, self.score2

    def is_collision(self, pt, snake_num):
        # hits boundary
        if self.is_bounds:
            if (
                pt.x > self.w - BLOCK_SIZE
                or pt.x < 0
                or pt.y > self.h - BLOCK_SIZE
                or pt.y < 0
            ):
                return 1

        # hits self or the body of the other snake
        if pt in self.snake1[1:] or pt in self.snake2[1:]:
            return 2

        # hits the head of the other snake
        if snake_num == 1 and pt == self.snake2[0]:
            return 3

        # hits the nead of the other snake
        if snake_num == 2 and pt == self.snake1[0]:
            return 3

        return 0

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake1:
            pygame.draw.rect(
                self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        for pt in self.snake2:
            pygame.draw.rect(
                self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display, GREEN2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
            )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(f"Score: {self.score1} - {self.score2}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move1(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction1)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction1 = new_dir

        x = self.head1.x
        y = self.head1.y

        if self.is_bounds:
            if self.direction1 == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction1 == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction1 == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction1 == Direction.UP:
                y -= BLOCK_SIZE
        else:
            if self.direction1 == Direction.RIGHT:
                x = (x + BLOCK_SIZE) % self.w
            elif self.direction1 == Direction.LEFT:
                x = (x - BLOCK_SIZE) % self.w
            elif self.direction1 == Direction.DOWN:
                y = (y + BLOCK_SIZE) % self.h
            elif self.direction1 == Direction.UP:
                y = (y - BLOCK_SIZE) % self.h

        self.head1 = Point(x, y)

    def _move2(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction2)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction2 = new_dir

        x = self.head2.x
        y = self.head2.y
        
        if self.is_bounds:
            if self.direction2 == Direction.RIGHT:
                x += BLOCK_SIZE
            elif self.direction2 == Direction.LEFT:
                x -= BLOCK_SIZE
            elif self.direction2 == Direction.DOWN:
                y += BLOCK_SIZE
            elif self.direction2 == Direction.UP:
                y -= BLOCK_SIZE
        else:
            if self.direction2 == Direction.RIGHT:
                x = (x + BLOCK_SIZE) % self.w
            elif self.direction2 == Direction.LEFT:
                x = (x - BLOCK_SIZE) % self.w
            elif self.direction2 == Direction.DOWN:
                y = (y + BLOCK_SIZE) % self.h
            elif self.direction2 == Direction.UP:
                y = (y - BLOCK_SIZE) % self.h

        self.head2 = Point(x, y)
