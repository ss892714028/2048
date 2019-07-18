from random import randint, random
import numpy as np


class Game:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int)
        self.game_over = False
        self.new_board = np.zeros((4, 4), dtype=np.int)
        self.step = 0

    def fill_cell(self):
        i, j = (self.board == 0).nonzero()
        if i.size != 0:
            rnd = randint(0, i.size - 1)
            self.board[i[rnd], j[rnd]] = 2 * ((random() > .9) + 1)

    def move_left(self, col):
        new_col = np.zeros((4), dtype=col.dtype)
        j = 0
        previous = None
        for i in range(col.size):
            if col[i] != 0:  # number different from zero
                if previous is None:
                    previous = col[i]
                else:
                    if previous == col[i]:
                        new_col[j] = 2 * col[i]
                        j += 1
                        previous = None
                    else:
                        new_col[j] = previous
                        j += 1
                        previous = col[i]
        if previous is not None:
            new_col[j] = previous
        return new_col

    def move(self, direction):
        # 0: left, 1: up, 2: right, 3: down
        rotated_board = np.rot90(self.board, direction)
        cols = [rotated_board[i, :] for i in range(4)]
        new_board = np.array([self.move_left(col) for col in cols])
        return np.rot90(new_board, -direction)

    def is_game_over(self):
        temp = []
        for i in range(3):
            temp.append((self.board == self.move(i)).all())
        if False not in temp:
            self.game_over = True
        if self.game_over:
            print('game_over')

    def main_loop(self, direction):
        self.new_board = self.move(direction)
        moved = False
        if (self.new_board == self.board).all():

            # move is invalid
            moved = False
        else:
            moved = True
            self.step += 1
            self.board = self.new_board
            self.fill_cell()
        self.is_game_over()
        return (self.board)
