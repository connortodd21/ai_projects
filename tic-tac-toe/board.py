import numpy as np
import copy

ROWS = 3
COLUMNS = 3

class Board(object):

    def __init__(self, board=np.zeros((ROWS, COLUMNS))):
        self.board = board

    def copy(self):
        return Board(self.board)

    def set_board(self, board):
        self.board = board

    def move(self, position, player):
        self.board[position[0]][position[1]] = player

    def get_token(self, position):
        return self.board[position[0]][position[1]]

    def get_possible_actions(self):
        actions = []
        for i in range(ROWS):
            for j in range(COLUMNS):
                if self.board[i][j] == 0:
                    actions.append((i,j))

        return actions

    def get_successor_states(self, player):
        actions = self.get_possible_actions()
        board = copy.deepcopy(self.board)
        successor_states = []
        for action in actions:
            self.move(action, player)
            tempBoard = Board(self.board)
            successor_states.append((tempBoard.board, action))
            self.board = copy.deepcopy(board)
        
        return successor_states

    def is_win(self):
        for i in range(ROWS):
            if sum(self.board[i, :]) == 3:
                return 1
            if sum(self.board[i, :]) == -3:
                return -1
        # col
        for i in range(COLUMNS):
            if sum(self.board[:, i]) == 3:
                return 1
            if sum(self.board[:, i]) == -3:
                return -1
        # diagonal
        diag_sum1 = sum([self.board[i, i] for i in range(COLUMNS)])
        diag_sum2 = sum([self.board[i, COLUMNS - i - 1] for i in range(COLUMNS)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.get_possible_actions()) == 0:
            return 0
        # not end
        return None

    def __str__(self):
        return str(self.board[0]) + '\n' + str(self.board[1]) + '\n' + str(self.board[2])
