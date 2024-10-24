import configs
import numpy as np


def generate_random_movements(n):
    # Initialize the rotation queue
    rotation_queue = []

    # print("possible_moves")

    for _ in range(n):
        action = np.random.choice(list(configs.rubiks_moves.keys()))
        rotation_queue.append(configs.rubiks_moves[action])
    return rotation_queue
