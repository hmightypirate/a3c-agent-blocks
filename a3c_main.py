import argparse
import logging
import numpy as np
import multia3c_agent as LAU


def process_args(args, defaults, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--training', dest="training",
                        action="store_true", default=defaults.TRAINING,
                        help=('Whether to perform training or testing.' +
                              '(default: % (default)s'))

    parser.add_argument('--epochs', dest="epochs",
                        type=bool, default=defaults.EPOCHS,
                        help=('Number of epoch during training' +
                              '(defaults: % (default)s'))

    parser.add_argument('--max_steps', dest="max_steps",
                        type=bool, default=defaults.MAX_STEPS,
                        help=('Maximum number of steps(defaults:' +
                              '% (default)s'))

    parser.add_argument('--batch_size', dest="batch_size",
                        type=bool, default=defaults.BATCH_SIZE,
                        help=('Batch size (defaults: %(default)s'))

    parser.add_argument('--num_threads', dest="num_threads",
                        type=int, default=defaults.NUM_THREADS,
                        help=('Number of threads of the A3C agent.' +
                              '(default: % (default)s'))

    parser.add_argument('--game', dest="game",
                        type=str, default=defaults.GAME,
                        help=('Atari game(Gym notation).' +
                              '(default: % (default)s'))

    parser.add_argument('--resized_width', dest="resized_width",
                        type=int, default=defaults.RESIZED_WIDTH,
                        help=('Resized width of the input image' +
                              '(default: % (default)s'))

    parser.add_argument('--resized_height', dest="resized_height",
                        type=int, default=defaults.RESIZED_HEIGHT,
                        help=('Resized height of the input image' +
                              '(default: % (default)s'))

    parser.add_argument('--agent_history_length',
                        dest="agent_history_length",
                        type=int, default=defaults.AGENT_HISTORY_LENGTH,
                        help=('History length (default: %(default)s'))

    parser.add_argument('--checkpoint_interval', dest="checkpoint_interval",
                        type=int, default=defaults.CHECKPOINT_INTERVAL,
                        help=('Iterations between checkpoints of the models' +
                              '(default: % (default)s'))

    parser.add_argument('--deterministic', dest="deterministic",
                        action="store_true", default=defaults.DETERMINISTIC,
                        help=('Perform deterministic training(defaults:' +
                              '% (default)s'))

    parser.add_argument('--sample_argmax', dest="sample_argmax",
                        action='store_true', default=defaults.SAMPLE_ARGMAX,
                        help=('Use argmax over the output distribution of' +
                              'actions(defaults: % (default)s'))

    parser.add_argument('--render_flag', dest="render_flag",
                        type=bool, default=defaults.RENDER_FLAG,
                        help=('True if the model should render the game' +
                              'during evaluations(defaults: % (default)s'))

    parser.add_argument('--results_file', dest="results_file",
                        type=str, default=defaults.RESULTS_FILE,
                        help=('File in which to store the results.' +
                              '(default: % (default)s'))

    parser.add_argument('--num_steps_eval', dest="num_steps_eval",
                        type=int, default=defaults.NUM_STEPS_EVAL,
                        help=('Num of evaluation steps(default:' +
                              '% (default)s'))

    parser.add_argument('--num_reps_eval', dest="num_reps_eval",
                        type=int, default=defaults.NUM_REPS_EVAL,
                        help=('Num of repetitions of an evaluation' +
                              '(default: % (default)s'))

    parser.add_argument('--learning_rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help=('Learning rate in optimization(default:' +
                              '% (default)s'))

    parser.add_argument('--gradient_clipping', dest="gradient_clipping",
                        type=float, default=defaults.GRADIENT_CLIPPING,
                        help=('Gradient Clipping (default: %(default)s'))

    parser.add_argument('--load_file', dest="load_file", default=None,
                        type=str, help=('Load pretrained model'))

    parser.add_argument('--gamma_rate', type=float, dest="gamma_rate",
                        default=defaults.GAMMA_RATE,
                        help=('Gamma rate propagating rewards backward'))
    
    parser.add_argument('--a3c_lstm', action="store_true", dest="a3c_lstm",
                        default=defaults.A3C_LSTM,
                        help=('Use A3C-LSTM model instead of A3C-FF ' +
                              '(default: %(default)s'))
                        
    parser.add_argument('--lstm_output_units', type=int,
                        dest="lstm_output_units",
                        default=defaults.LSTM_OUTPUT_UNITS,
                        help=('Number of output units in A3C-LSTM agent' +
                              ' (default: % (default)s'))
                        
    parameters = parser.parse_args(args)

    return parameters


def launch(args, defaults, description):
    """ Basic launch functionality """

    logging.basicConfig(level=logging.INFO)
    parameters = process_args(args, defaults, description)

    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    agent = LAU.MultiA3CTrainStream(
        rng=rng,
        epochs=parameters.epochs,
        max_steps=parameters.max_steps,
        batch_size=parameters.batch_size,
        game=parameters.game,
        num_threads=parameters.num_threads,
        resized_width=parameters.resized_width,
        resized_height=parameters.resized_height,
        agent_history_length=parameters.agent_history_length,
        checkpoint_interval=parameters.checkpoint_interval,
        training_flag=parameters.training,
        sample_argmax=parameters.sample_argmax,
        render_flag=parameters.render_flag,
        results_file=parameters.results_file,
        num_steps_eval=parameters.num_steps_eval,
        num_reps_eval=parameters.num_reps_eval,
        gamma_rate=parameters.gamma_rate,
        learning_rate=parameters.learning_rate,
        gradient_clipping=parameters.gradient_clipping,
        model_file=parameters.load_file,
        a3c_lstm=parameters.a3c_lstm,
        lstm_output_units=parameters.lstm_output_units)

    logging.info("Num Actors {}".format(parameters.num_threads))
    agent.execute()


if __name__ == '__main__':
    pass
