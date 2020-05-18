from agent import Agent
from board import Board
from util import get_args
from human import Human

import sys
import copy
import time

def give_reward(agent, result, opponent, move_list):
    if result == agent.player:
        agent.back_propogate(10, move_list)
        opponent.back_propogate(-10, move_list)
    elif result == opponent.player:
        agent.back_propogate(-10, move_list)
        opponent.back_propogate(10, move_list)
    else:
        agent.back_propogate(5, move_list)
        opponent.back_propogate(5, move_list)

def give_reward_from_human(agent, result, move_list):
    if result == agent.player:
        agent.back_propogate(10, move_list)
    elif result == (agent.player * -1):
        agent.back_propogate(-10, move_list)
    else:
        agent.back_propogate(-1, move_list)

def self_play(agent1, agent2):
    board = Board()
    agent1.set_board(copy.deepcopy(board))
    agent2.set_board(copy.deepcopy(board))
    move_list = []
    move_list.append(copy.deepcopy(board.board))
    while(board.is_win() is None):
        agent1move = agent1.make_move(False)
        board = copy.deepcopy(agent1.board)
        if agent1move == 0 or len(board.get_possible_actions()) == 0 or board.is_win() is not None:
            agent2.set_board(copy.deepcopy(board))
            move_list.append(copy.deepcopy(board.board))
            break

        agent2.set_board(copy.deepcopy(board))
        move_list.append(copy.deepcopy(board.board))
        agent2move = agent2.make_move(False)
        board = copy.deepcopy(agent2.board)
        if agent2move == 0 or len(board.get_possible_actions()) == 0 or board.is_win() is not None:
            move_list.append(copy.deepcopy(board.board))
            agent1.set_board(copy.deepcopy(board))
            break

        move_list.append(copy.deepcopy(board.board))
        agent1.set_board(copy.deepcopy(board))

    outcome = board.is_win()
    give_reward(agent1, outcome, agent2, move_list)
    print(outcome)

def play_human_vs_ai(xplayer, oplayer, human_token):
    board = Board()
    xplayer.set_board(copy.deepcopy(board))
    oplayer.set_board(copy.deepcopy(board))
    move_list = []
    move_list.append(copy.deepcopy(board.board))
    while(board.is_win() is None):
        # human is X
        time.sleep(.5)
        print("\n")
        print(board)
        if human_token == 1:
            move = xplayer.cli_ask_for_move()
            xplayer.make_move(move)
            board = copy.deepcopy(xplayer.board)
            if len(board.get_possible_actions()) == 0 or board.is_win() is not None:
                oplayer.set_board(copy.deepcopy(board))
                move_list.append(copy.deepcopy(board.board))
                break

            oplayer.set_board(copy.deepcopy(board))
            move_list.append(copy.deepcopy(board.board))
            print("\n")
            print(board)
            aimove = oplayer.make_move(True)
            board = copy.deepcopy(oplayer.board)
            if aimove == 0 or len(board.get_possible_actions()) == 0 or board.is_win() is not None:
                move_list.append(copy.deepcopy(board.board))
                oplayer.set_board(copy.deepcopy(board))
                break

            move_list.append(copy.deepcopy(board.board))
            xplayer.set_board(copy.deepcopy(board))

        else:
            aimove = xplayer.make_move(True)
            board = copy.deepcopy(xplayer.board)
            if aimove == 0 or len(board.get_possible_actions()) == 0 or board.is_win() is not None:
                move_list.append(copy.deepcopy(board.board))
                xplayer.set_board(copy.deepcopy(board))
                break
            print("\n", board)
            move_list.append(copy.deepcopy(board.board))
            oplayer.set_board(copy.deepcopy(board))
            move = oplayer.cli_ask_for_move()
            oplayer.make_move(move)
            board = copy.deepcopy(oplayer.board)
            if len(board.get_possible_actions()) == 0 or board.is_win() is not None:
                xplayer.set_board(copy.deepcopy(board))
                move_list.append(copy.deepcopy(board.board))
                break

            xplayer.set_board(copy.deepcopy(board))
            move_list.append(copy.deepcopy(board.board))

    outcome = board.is_win()
    print("\n")
    print(board)
    if outcome == human_token:
        print("Congrats! You win!")
    elif outcome == 0:
        print("Game is a draw")
    else:
        print("You lose")
    if human_token == 1:
        give_reward_from_human(oplayer, outcome, move_list)
    else:
        give_reward_from_human(xplayer, outcome, move_list)
        

def main():
    epsilon, discount, alpha, iterations, selfPlay, readValues = get_args(sys.argv)
    if selfPlay == True:
        agent1 = Agent(1, epsilon, discount, alpha)
        agent2 = Agent(-1, epsilon, discount, alpha)
        print("Beginning self play. Corresponding state values will be stored in agent1_values.txt and agent2_values.txt")
        for i in range(iterations):
            print("Iteration %d..." %(i))
            self_play(agent1, agent2)
        
        agent1.write_qvalues('agent1_values.txt')
        agent2.write_qvalues('agent2_values.txt')

    elif readValues == True:
        token = 0
        ai = 0
        while(True):
            token = input("What piece would you like to be (X/O)")
            if token == "X" or token == "O":
                break
        if token == "X":
            token = 1
            ai = Agent(token*-1, epsilon=.2, discount=.7, alpha=.7, readValues=True, file="./agent2_values.txt")
        else:
            token = -1
            ai = Agent(token*-1, epsilon=.2, discount=.7, alpha=.7, readValues=True, file="./agent1_values.txt")
        human = Human(token)
        if token == 1:
            # human is X
            play_human_vs_ai(human, ai, token)
            ai.write_qvalues("agent2_values.txt")
        else:
            play_human_vs_ai(ai, human, token)
            ai.write_qvalues("agent1_values.txt")

if __name__ == '__main__':
	main()
    