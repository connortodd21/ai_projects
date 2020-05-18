import random
import sys
import inspect

def __init__():
    return 

def flip_coin(prob):
    r = random.random()
    return r > float(prob)

def list_to_2dlist(list):
    temp = []
    ret = []
    for i in range(3):
        temp.append(list[i])
    ret.append(temp)
    temp = []
    for i in range(3,6):
        temp.append(list[i])
    ret.append(temp)
    temp = []
    for i in range(6,9):
        temp.append(list[i])
    ret.append(temp)
    return ret

def get_args(args):
    epsilon = 0
    discount = 0
    alpha = 0
    iterations = 100
    i = 0
    selfPlay = False
    readValues = False
    try:
        for arg in args:
            if arg == "-e":
                epsilon = args[i+1]
            elif arg == "-d":
                discount = args[i+1]
            elif arg == "-a":
                alpha = args[i+1]
            elif arg == '-s':
                if readValues != True:
                    selfPlay = True
                else:
                    raise_incorrect_arguments()
            elif arg == "-p":
                if selfPlay != True:
                    readValues = True
                else:
                    raise_incorrect_arguments()
            elif arg == "-h":
                    help()
            elif arg == "-i":
                iterations = args[i+1]
            i+=1
    except Exception:
        raise_incorrect_arguments()

    return (epsilon, discount, alpha, int(iterations), selfPlay, readValues)

def help():
    print("** For general use ***")
    print("-h   |   help")
    print("-p   |   Play a game againt the AI")
    print("*** For self-play ***")
    print("-s   |   self-play between two agents to learn proper policy")
    print("-e   |   specify the value of epsilon 0<=e<=1")
    print("-d   |   specify the value of gamma (discount) 0<=d<=1")
    print("-i   |   specify the numeber of iterations for self play. Defaults to 100")
    print("-a   |   specify the value of alpha (learning rate) 0<=a<=1\n")
    sys.exit(1)

def raise_incorrect_arguments():
    print("\nUsage: python3 game.py [-h] [-r] [-s] [-e] [-d] [-a]\n")
    help()

def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)
