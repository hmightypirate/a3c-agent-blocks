#! /user/bin/env python

import a3c_main
import sys


class Defaults:

    #------------------------
    # Experiment Parameters
    #------------------------
    TRAINING = False
    EPOCHS = 100  # FIXME: not used?
    MAX_STEPS = 160000000
    BATCH_SIZE = 5
    DETERMINISTIC = False
    SAMPLE_ARGMAX = False
    NUM_REPS_EVAL = 10
    NUM_STEPS_EVAL = 10000
    RESULTS_FILE = "log/results"
    LEARNING_RATE = 0.0007
    GRADIENT_CLIPPING = None
    RENDER_FLAG = True

    #------------------------
    # GYM Parameters
    #------------------------
    GAME = 'Breakout-v0'

    #-----------------------
    # A3C Parameters
    #-----------------------
    NUM_THREADS = 8  # 16
    RESIZED_WIDTH = 84
    GAMMA_RATE = 0.99
    RESIZED_HEIGHT = 84
    AGENT_HISTORY_LENGTH = 4
    CHECKPOINT_INTERVAL = 50000
    EPSILON_MIN = 0.  # FIXME: not used?
    EPSILON_INIT_MIN = 0.  # FIXME: not used?
    EPSILON_MAX = 0.  # FIXME: not used
    EPSILON_DECAY = 0.000001  # FIXME: not used


if __name__ == "__main__":
    a3c_main.launch(sys.argv[1:], Defaults, __doc__)
