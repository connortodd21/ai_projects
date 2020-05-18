import random
import pickle

ROWS = 3
COLUMNS = 3

class Human:

    def __init__(self, token):
        self.token = token

    def set_board(self, board):
        self.board = board

    def make_move(self, position):
        self.board.board[position[0]][position[1]] = self.token

    def cli_ask_for_move(self):
        while(True):
            try:
                while(True):
                    move = input("Please enter the x and y coordinates for your move as follows x,y: ")
                    xcoord = int(move[0])
                    ycoord = int(move[2])
                    if self.board.board[xcoord][ycoord] != 0:
                        print("This spot is already taken, try again")
                    else:
                        return (xcoord, ycoord)
            except Exception:
                print("Input error, please try again")

    