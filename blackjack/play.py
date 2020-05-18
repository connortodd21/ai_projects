import random
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

player_stop_scores = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

class Blackjack:

    def __init__(self, stop_score):
        self.values = {}
        self.history = []
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.stop_score = stop_score

    def hit(self):
        possibleCards = list(range(1,11))
        possibleCards.append(10)
        possibleCards.append(10)
        possibleCards.append(10)
        return random.choice(possibleCards)

    def dealer(self, score, usable_ace):
        # Bust unless we have an ace we can use as a 1
        if score > 21:
            if usable_ace:
                score -= 10
                usable_ace = False

            else:
                return (score, usable_ace, 1)

        # 17, Must stop
        if score > 17:
            return(score, usable_ace, 1)

        else:
            card = self.hit()
            if card == 1:
                if score <= 10:
                    # Use the ace as an 11
                    return (score + 11, True, 0)
                else:
                    # Use the ace as a 1
                    return (score + 1, usable_ace, 0)
            else:
                return (score + card, usable_ace, 0)

    def player(self, score, usable_ace):
        # Bust unless we have an ace we can use as a 1
        if score > 21:
            if usable_ace:
                score -= 10
                usable_ace = False

            else:
                return (score, usable_ace, 1)
        
        if score > self.stop_score:
            return(score, usable_ace, 1)
        
        else:
            card = self.hit()
            if card == 1:
                if score <= 10:
                    # Use the ace as an 11
                    return (score + 11, True, 0)
                else:
                    # Use the ace as a 1
                    return (score + 1, usable_ace, 0)
            else:
                return (score + card, usable_ace, 0)

    def update_weights(self, player_score, dealer_score):
        last_state = self.history[-1]
        if player_score > 21:
            if dealer_score > 21:
                # draw
                self.draws += 1
            else:
                self.values[last_state] -= 1
                self.losses += 1
        else:
            if dealer_score > 21:
                self.values[last_state] += 1
                self.wins += 1
            else:
                if player_score < dealer_score:
                    self.values[last_state] -= 1
                    self.losses += 1
                elif player_score > dealer_score:
                    self.values[last_state] += 1
                    self.wins += 1
                else:
                    # draw
                    self.draws += 1

    def simulate(self, iterations=100):
        for i in range(iterations):
            if i % 1000 == 0:
                print("Iteration %d..." % i)
            dealer_score = 0
            show_card = 0
            player_score = 0
            dealer_score += self.hit()
            show_card = dealer_score
            dealer_score += self.hit()

            usable_ace = False
            game_over = 0
            # player
            while True:
                player_score, usable_ace, game_over = self.player(player_score, usable_ace)
                if game_over == 1:
                    break

                if (player_score >= 12) and (player_score <= 21):
                    self.history.append((player_score, show_card, usable_ace))
                
            # dealer
            while True:
                dealer_score, usable_ace, game_over = self.dealer(dealer_score, usable_ace)
                if game_over == 1:
                    break
            
            for state in self.history:
                if state not in self.values.keys():
                    self.values[state] = 0
                
            self.update_weights(player_score, dealer_score)

def plot_weights(blackjack):
    print("Plots ----------------")
    usable_ace = {}
    nonusable_ace = {}

    for k, v in blackjack.values.items():
        if k[2]:
            usable_ace[k] = v
        else:
            nonusable_ace[k] = v

    fig = plt.figure(figsize=[15, 6])

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x1 = [k[1] for k in usable_ace.keys()]
    y1 = [k[0] for k in usable_ace.keys()]
    z1 = [v for v in usable_ace.values()]
    ax1.scatter(x1, y1, z1)

    ax1.set_title("usable ace")
    ax1.set_xlabel("dealer showing")
    ax1.set_ylabel("player sum")
    ax1.set_zlabel("reward")

    x2 = [k[1] for k in nonusable_ace.keys()]
    y2 = [k[0] for k in nonusable_ace.keys()]
    z2 = [v for v in nonusable_ace.values()]
    ax2.scatter(x2, y2, z2)

    ax2.set_title("non-usable ace")
    ax2.set_xlabel("dealer showing")
    ax2.set_ylabel("player sum")
    ax2.set_zlabel("reward")

    plt.show()

def main():
    iterations = 100
    try:
        if len(sys.argv) == 3:
            if sys.argv[1] == "-i":
                iterations = int(sys.argv[2])
    except Exception:
        print("Usage: python3 play.py [-i]")
        print("-i   |   specify the number of iterations for training (i is an int)")
        sys.exit(1)

    max_wins = 0
    losses = 0
    draws = 0
    best_score = 0
    loss_pct = 1.0
    win_pct = 1.0
    best_loss_pct_score = 0
    print("----------------------------------------------------------")
    for score in player_stop_scores:
        game = Blackjack(score)
        game.simulate(iterations)
        # plot_weights(game)
        if game.wins > max_wins:
            max_wins = game.wins
            best_score = score
            losses = game.losses
            draws = game.draws

        if (game.losses) / (game.losses + game.draws + game.wins) < loss_pct:
            loss_pct = float((game.losses)) / float((game.losses + game.draws + game.wins))
            win_pct = float((game.wins)) / float((game.losses + game.draws + game.losses))
            best_loss_pct_score = score
        print("Card to stop hitting on: %d" %(score))
        print("Wins %d" %(game.wins))
        print("Draws %d" %(game.draws))
        print("Losses %d\n" %(game.losses))
        print("----------------------------------------------------------")

    print("Best card to stop hitting on was %d with %d wins, %d draws, and %d losses" %(best_score, max_wins, draws, losses))
    print("Stop card with the lowest loss percentage was %d with %f loss percentage and %f win percentage" %(best_loss_pct_score, loss_pct, win_pct))

if __name__ == "__main__":
    main()
