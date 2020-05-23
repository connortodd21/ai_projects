import json
import random

class Agent:

    def __init__(self, discount=.9, alpha=.7, epsilon=.3):
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.its = 0
        """
            Qvalues are stored here in a state, value pair
            State is defined as (x,y,v,p)
            x   :   x-distance to next pipe
            y   :   y-distance to next pipe
            v   :   velocity
            o   :   open space in the next pipe
            Value is defined as (value_if_no_flap, value_if_flap, number_of_updates)
        """
        self.qvalues = {}
        self.read_qvalues()
        """
            This will be a different way of solving the problem
            This dict will only contain the current qvalue as the value to the state key and will rely on looking at child values
        """
        self.qvalues_successors = {}
        """
            History is the move history
            Move is defined as (state, action, nextState)
        """
        self.MIN_VALUE = -9999999
        self.move_history = []
        self.num_games = 0
        self.last_state = "0_0_0_0"
        self.qvalues[self.last_state] = [0, 0, 0]
        self.last_action = 0

    def get_reward(self, alive_status):
        if alive_status == True:
            return 0
        return -1000

    def save_qvalues(self):
        if len(self.move_history) > 6_000_000:
            history = list(reversed(self.move_history[:5_000_000]))
            for move in history:
                state, action, nextState = move
                reward = self.get_reward(True)
                value = self.qvalues[state][action]
                self.qvalues[state][action] = ((1 - self.alpha) * value) + (self.alpha * (reward + self.discount * max(self.qvalues[nextState][0:2])))
            
            self.moves = self.moves[5_000_000:]

    def update_qvalues(self):
        history = list(reversed(self.move_history))
        reward = 0
        moves_until_death = 0
        high_death_flag = True if len(history) == 0 or int(history[0][2].split("_")[1]) > 120 else False
        last_move_is_flap = True
        self.its+=1

        if self.its == 5:
            print(history)

        for move in history:
            state = move[0]
            action = move[1]
            nextState = move[2]
            reward = 0

            if self.its == 5:
                print(action)
                print(self.qvalues[state])

            self.qvalues[state][2] += 1

            if moves_until_death <= 4:
                reward = self.get_reward(False)
                if action == 1:
                    last_move_is_flap = False
            elif (high_death_flag or last_move_is_flap) and action == 1:
                high_death_flag = False
                last_move_is_flap = False
                reward = self.get_reward(False)
            else:
                reward = self.get_reward(True)

            if self.its == 5:
                print(reward)
            
            moves_until_death += 1
            value = self.qvalues[state][action]
            self.qvalues[state][action] = ((1 - self.alpha) * value) + (self.alpha * (reward + self.discount * max(self.qvalues[nextState][0:2])))
            if self.its == 5:
                print(self.qvalues[state])

        self.move_history = []
        self.num_games += 1

    def flip_coin(self, prob):
        r = random.random()
        return r > float(prob)
    
    def epsilon_greedy(self, state, bestMove):
        if not bestMove:
            move = None
            randomMoveProb = self.flip_coin(self.epsilon)
            if randomMoveProb > (1-randomMoveProb):
                move = self.make_move(state)
                return move
            else:
                move = random.choice([0,1])
                return move
        else:
            move = self.make_move(state)
            return move

    def make_move_successors(self, state, successors):
        self.move_history.append((self.last_state, self.last_action, state))
        move = 0
        maxValue = self.MIN_VALUE
        for x,y,vel,pipe,act in successors:
            state = self.format_state(x,y,vel,pipe)
            if self.qvalues_successors[state] > maxValue:
                move = act

        self.last_action = move
        self.last_state = state
        return move

    def make_move(self, state):
        self.move_history.append((self.last_state, self.last_action, state))
        self.save_qvalues()
        move = 0
        if self.qvalues[state][1] > self.qvalues[state][0]:
            move = 1

        self.last_action = move
        self.last_state = state
        return move

    def write_qvalues(self):
        fil = open("qvalues.json", "w")
        json.dump(self.qvalues, fil)
        fil.close()

    def read_qvalues(self):
        self.qvalues = {}
        try:
            fil = open("qvalues.json", "r")
        except IOError:
            return
        self.qvalues = json.load(fil)
        fil.close()

    def game_over(self):
        self.update_qvalues()
        self.last_state = "0_0_0_0"
        self.last_action = 0

    def format_state(self, playerx, playery, velocity, pipe):
        """
        format:
            x0_y0_v_y1
        (x, y): coordinates of player (top left point)
        x0: diff of x to pipe0, [-50, ...]
        y0: diff of y to pipe0
        v: current velocity
        y1: diff of y to pipe1
        """
        pipe0 = pipe[0]
        pipe1 = pipe[1]
        if playerx - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]

        x0 = pipe0["x"] - playerx
        y0 = pipe0["y"] - playery
    
        if -50 < x0 <= 0:  
            y1 = pipe1["y"] - playery
        else:
            y1 = 0

        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)
        else:
            x0 = int(x0) - (int(x0) % 70)

        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:
            y0 = int(y0) - (int(y0) % 60)

        #x1 = int(x1) - (int(x1) % 10)
        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)

        state = str(int(x0)) + "_" + str(int(y0)) + "_" + str(int(velocity)) + "_" + str(int(y1))
        if self.qvalues.get(state) == None:
             self.qvalues[state] = [0, 0, 0]
        return state

