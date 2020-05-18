from util import flip_coin
from util import list_to_2dlist
from board import Board

import random
import numpy as np

ROWS = 3
COLUMNS = 3

class Agent:

    def __init__(self, token, epsilon=.5, discount=.7, alpha=.2, readValues=False, file=""):
        self.player = token
        self.QValues = {} # key = board (array), value = qvalue
        self.MIN_VALUE = -999
        self.discount = float(discount)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        if readValues == True:
            self.read_qvalues(file)

    def set_board(self, board):
        self.board = board

    def hash(self, board):
        boardHash = str(board.reshape(ROWS * COLUMNS))
        boardHash = boardHash.replace(".", ",")
        return boardHash

    def get_qvalue(self, state):
        if self.hash(state) in self.QValues.keys():
            return self.QValues[self.hash(state)]

        return 0

    def find_move_from_qvalues(self, currentState):
        successorStates = currentState.get_successor_states(self.player)
        maxAction = 0
        maxValue = self.MIN_VALUE
        for state, action in successorStates:
            value = self.get_qvalue(state)
            if value > maxValue:
                maxValue = value
                maxAction = action
        
        return maxAction

    def back_propogate(self, reward, move_list):
        for state in reversed(move_list):
            value = self.get_qvalue(state)
            newValue = value + self.alpha * (self.discount * reward - value)
            self.QValues[self.hash(state)] = newValue
            reward = self.get_qvalue(state)

    def epsilon_greedy(self, state, bestMove):
        if not bestMove:
            legalActions = self.board.get_possible_actions()
            move = None
            randomMoveProb = flip_coin(self.epsilon)
            if len(legalActions) > 0:
                if randomMoveProb > (1-randomMoveProb):
                    move = self.find_move_from_qvalues(state)
                    return move
                else:
                    move = random.choice(legalActions)
                    return move
            return None
        else:
            legalActions = self.board.get_possible_actions()
            move = None
            if len(legalActions) > 0:
                move = self.find_move_from_qvalues(state)
                return move
            return None

    def make_move(self, bestMove=True):
        move = self.epsilon_greedy(self.board, bestMove)
        if move is not None:
            self.board.move(move, self.player)
            return 1
        else:
            return 0

    def write_qvalues(self, file):
        with open(file, 'w') as f:
            for key, value in self.QValues.items():
                f.write('%s:%s\n' % (key, value))

    def read_qvalues(self, file):
        data = {}
        with open(file, 'r') as raw_data:
            for item in raw_data:
                if ':' in item:
                    key,value = item.split(':', 1)
                    key = key[1:-2]
                    key = key.replace(" ", "").split(",")
                    key = list(map(float, key))
                    board = list_to_2dlist(key)
                    board = np.array(board)
                    data[self.hash(board)]=float(value)
                else:
                    pass # deal with bad lines of text here
        self.QValues = data

    def display_qvalues(self):
        for key in self.QValues.keys():
            print(key, end=" ")
            print(self.QValues[key], end=" ")
            print("\n")
