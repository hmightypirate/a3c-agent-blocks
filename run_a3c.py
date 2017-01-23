#! /user/bin/env python

import a3c_main
import sys


class Defaults:

    #------------------------
    # Experiment Parameters
    #------------------------
    TRAINING = False  # testing by default
    EPOCHS = 100  # FIXME: not used
    MAX_STEPS = 160000000
    BATCH_SIZE = 32
    DETERMINISTIC = False  # use deterministic seed
    SAMPLE_ARGMAX = False  # False means use exploration
    NUM_REPS_EVAL = 10
    NUM_STEPS_EVAL = 10000
    # folder and prefix file
    # (the folder must exist before launching the code)
    RESULTS_FILE = "log/results"
    LEARNING_RATE = 0.0007
    GRADIENT_CLIPPING = None
    RENDER_FLAG = True  # only whilst evaluating the model

    #------------------------
    # GYM Parameters
    #------------------------
    GAME = 'Breakout-v0'

    #-----------------------
    # A3C Parameters
    #-----------------------
    NUM_THREADS = 16
    RESIZED_WIDTH = 84
    GAMMA_RATE = 0.99
    RESIZED_HEIGHT = 84
    AGENT_HISTORY_LENGTH = 4
    CHECKPOINT_INTERVAL = 50000
    A3C_LSTM = False
    LSTM_OUTPUT_UNITS = 256


if __name__ == "__main__":
    a3c_main.launch(sys.argv[1:], Defaults, __doc__)
