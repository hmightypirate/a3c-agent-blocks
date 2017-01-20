import gym
import gym.wrappers
from theano import tensor as T
import threading
import time
import numpy as np
import logging
from atari_environment import AtariEnvironment
from collections import OrderedDict
from blocks import serialization

import network as A3C

logger = logging.getLogger(__name__)


def sample_policy_action(num_actions, probs, rng):
    """ Sample an action using a prob. distribution

    Parameters
    ----------
    num_actions : int
    number of available actions

    probs : list
    list of float with the probability of each action

    """
    probs = probs - np.finfo(np.float32).epsneg
    histogram = rng.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index


def sample_argmax_action(num_actions, probs, rng):
    """ Pick the argmax action

    Parameters
    ----------
    num_actions : int
    number of available actions

    probs : list
    list of float with the probability of each action

    """
    action_index = int(np.argmax(probs))
    return action_index


class Common_Model(object):
    """ A container class for the shared model

    Parameters
    ----------
    rng : numpy.random
    game : string
        gym name of the game
    model: instance of :class:`.Model`
        the shared model with the cost application
    algorithm: instance of :class: `theano.function`
        gradient of the cost function of the model
    policy_network: intance of :class:`theano.function`
        the policy function of the shared model
    value_network: instance of :class:`theano.function`
        the value function of the shared model
    monitor_env: instance of class `gym.environment`
        a gym environment
    resized_width: int
        the width of the images that will be used by the agent
    resized_height: int
        the height of the images that will be used by the agent
    agent_history_length: int
        number of past frames that will be used by the agent
    max_steps: int
        maximum number of steps the agent will play during training
    render_flag: bool
        True to render the screen whilst playing
    results_file: str
        prefix path for storing the results
    num_steps_eval: int
        maximum number of steps the agent will play during evaluation
    sample_argmax: bool
        True if the action of max prob should be chosen or False to choose
        a multinomial sampling strategy
    """

    def __init__(self, rng, game, model, algorithm, policy_network,
                 value_network, monitor_env,
                 resized_width, resized_height, agent_history_length,
                 max_steps, render_flag=False, results_file=False,
                 num_steps_eval=100, sample_argmax=False):
        """ The common model """

        self.rng = rng
        self.game = game
        self.model = model
        self.curr_steps = 0
        self.algorithm = algorithm
        self.policy_network = policy_network
        self.value_network = value_network
        self.monitor_env = monitor_env
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.env = AtariEnvironment(
            gym_env=self.monitor_env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length)

        self.num_steps_eval = num_steps_eval
        self.max_steps = max_steps
        self.render_flag = render_flag
        self.results_file = results_file

        # Sets the sampling strategy
        if sample_argmax:
            self._sample_function = sample_argmax_action
        else:
            self._sample_function = sample_policy_action

        self.curr_it = 0

    def reset_environment(self):
        """ Reloads an environment
        """
        self.env = AtariEnvironment(
            gym_env=self.monitor_env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length)


class Common_Model_Wrapper(object):
    """ A wrapper class to store the model shared by all the learning agents

    It implements the lock for concurrent thread access to the shared model

    Parameters
    ----------
    rng : numpy.random
    game : string
        gym name of the game
    model: instance of :class:`.Model`
        the shared model with the cost application
    algorithm: instance of :class: `theano.function`
        gradient of the cost function of the model
    policy_network: intance of :class:`theano.function`
        the policy function of the shared model
    value_network: instance of :class:`theano.function`
        the value function of the shared model
    monitor_env: instance of class `gym.environment`
        a gym environment
    resized_width: int
        the width of the images that will be used by the agent
    resized_height: int
        the height of the images that will be used by the agent
    agent_history_length: int
        number of past frames that will be used by the agent
    max_steps: int
        maximum number of steps the agent will play during training
    render_flag: bool
        True to render the screen whilst playing
    results_file: str
        prefix path for storing the results
    num_steps_eval: int
        maximum number of steps the agent will play during evaluation
    sample_argmax: bool
        True if the action of max prob should be chosen or False to choose
        a multinomial sampling strategy

    """

    def __init__(self, rng, game, model, algorithm, policy_network,
                 value_network, monitor_env, resized_width, resized_height,
                 agent_history_length, max_steps, render_flag=False,
                 results_file=False, num_steps_eval=100, sample_argmax=False):

        self.common_model = Common_Model(rng, game, model, algorithm,
                                         policy_network,
                                         value_network, monitor_env,
                                         resized_width,
                                         resized_height,
                                         agent_history_length,
                                         max_steps, render_flag,
                                         results_file,
                                         num_steps_eval, sample_argmax)

        self.lock = threading.Lock()

    def save_model(self, save_file):
        """ Dump the current shared model to a file

        Parameters
        ----------
        save_file: str
            path in which to save the model

        """
        self.lock.acquire()
        with open(save_file, "wb+") as dst:
            serialization.dump(self.common_model, dst,
                               parameters=self.common_model.model.parameters)
        self.lock.release()

    def load_model(self, load_file):
        """ Loading model parameters

        Parameters
        ----------
        load_file: str
            path to a checkpoint of a model

        """
        self.lock.acquire()
        with open(load_file, 'rb') as src:
            parameters = serialization.load_parameters(src)
            self.common_model.model.set_parameter_values(
                parameters.get_values())
        self.lock.release()

    def update_cum_gradients(self, init_state, last_state, batch_size=1,
                             stats_flag=False):
        """ Obtain the cum gradient of the iteration and updates the model

        Parameters
        ----------
        init_state : OrderedDict
           parameters of the model at the beginning of an iteration
        last_state : OrderedDict
          parameters of the model at the end on an iteration
        batch_size: int
           size of the current batch
        stats_flag: bool
          whether to show current update stats on screen

        """

        # Obtain the acc gradient information
        new_update = OrderedDict()

        for kk in init_state:
            if kk in last_state:
                new_update[kk] = (init_state[kk] - last_state[kk]) * batch_size
            else:
                logger.error("{} is not part of the update ".format(kk))

        if (stats_flag):
            for kk in self.common_model.model.get_parameter_dict():
                logger.info(("Before Update {} with Mean {} Current {}" +
                             "Shape {}").format(
                    kk,
                    np.mean(new_update[kk]),
                    np.mean(
                        self.common_model.model.get_parameter_dict()[
                            kk].get_value()),
                    np.shape(
                        self.common_model.model.get_parameter_dict()[
                            kk].get_value())))
        try:
            # Acquiring the lock and perform update
            self.lock.acquire()
            self.common_model.algorithm.process_batch(new_update)
            self.lock.release()

            if (stats_flag):
                logger.info(("Before Update {} with Mean {} Current {}" +
                             "Shape {}").format(
                    kk,
                    np.mean(new_update[kk]),
                    np.mean(
                        self.common_model.model.get_parameter_dict()[
                            kk].get_value()),
                    np.shape(
                        self.common_model.model.get_parameter_dict()[
                            kk].get_value())))

        finally:
            pass

    def synchronize_model(self,  agent_model):
        """ Update the parameters of the learning agent with current
        shared model parameters

        Parameters
        ----------
        agent_model : instance of :class: `~blocks.model.Model`

        """

        self.lock.acquire()  # FIXME: remove
        try:
            agent_model.set_parameter_values(
                self.common_model.model.get_parameter_values())

        finally:
            self.lock.release()  # FIXME: remove

    def perform_evaluation(self, num_reps, experiment_name,
                           eval_it, do_gym_eval=False):
        """ Perform an evaluation of the current model

        Parameters
        ----------
        num_reps: int
          number of times the test experiment will be repeated
        experiment_name: str
          name of the experiment
        eval_it : int
          number of evaluation iterations
        do_gym_eval: bool
          if gym statistics should be gathered during evaluation

        """

        # Capturing the lock during all the evaluation period
        self.common_model.reset_environment()

        try:
            self.lock.acquire()
            print "EVAL: GOING TO EVAL "
            if do_gym_eval:
                self.common_model.monitor_env.monitor.start(
                    "{}{}_{}/eval".format(
                        self.common_model.results_file,
                        experiment_name,
                        eval_it))
            # Stat vars
            test_rewards = []
            test_ep_length = []
            test_value = []

            for i_episode in xrange(num_reps):
                s_t = self.common_model.env.get_initial_state()
                terminal = False
                ep_reward = 0
                ep_t = 0

                # Execute actions till achieving a finish state or the eval
                # period is over
                while not terminal and ep_t < self.common_model.num_steps_eval:
                    if self.common_model.render_flag and do_gym_eval:
                        self.common_model.monitor_env.render()

                    probs = self.common_model.policy_network([s_t])[0][0]
                    action_index = self.common_model._sample_function(len(
                        self.common_model.env.gym_actions), probs,
                        self.common_model.rng)

                    s_t1, r_t, terminal, info = self.common_model.env.step(
                        action_index)
                    s_t = s_t1
                    ep_reward += r_t
                    ep_t += 1

                    test_value.append(
                        self.common_model.value_network([s_t])[0][0])

                test_rewards.append(ep_reward)
                test_ep_length.append(ep_t)

            # TODO: save stats to file
            self._save_eval_to_file("{}{}".format(
                self.common_model.results_file,
                self.common_model.game),
                                    eval_it,
                                    test_rewards,
                                    test_ep_length,
                                    test_value)
            if do_gym_eval:
                self.common_model.monitor_env.close()
        except Exception:
            print "Exception whilst evaluating ", eval_it
        finally:
            self.lock.release()

    def _save_eval_to_file(self, eval_file, eval_it,
                           test_rewards,
                           test_ep_length,
                           test_value):
        """ Save stats to a file

        This function was intended for use at several timesteps
        during training.

        Parameters
        ----------
        eval_file: str
          path in which the stats will be appended
        eval_it : int
          iteration in which the eval is performed
        test_rewards: list of int
          rewards obained in the evaluation
        test_ep_length: list of int
          length of each episode
        test_value: list of float
          value at each point of an experiment
        """

        with open(eval_file, "a+") as f_eval:
            f_eval.write("{},{},{},{}\n".format(eval_it,
                                                np.mean(test_rewards),
                                                np.mean(test_ep_length),
                                                np.mean(test_value)))


def extract_params_from_model(common_net):
    """ Obtain the parameters of a model in an OrderedDict

    Parameters
    ----------
    common_net: instance of :class: `~blocks.model.Model`
      the model from where the parameters are extracted

    """
    params = OrderedDict()

    for kk in common_net.get_parameter_dict():
        params[kk] = common_net.get_parameter_dict()[kk].get_value()

    return params


class A3C_Agent(threading.Thread):
    """ This class implements the one thread agent of A3C

    Parameters
    ----------
    rng : numpy.random
    num : int
      id of the thread
    env : instance of :class: `gym.environment`
      gym environment of this thread
    batch_size: int
      maximum size of a training batch
    policy_network: intance of :class:`theano.function`
        the policy function of the worker
    value_network: instance of :class:`theano.function`
        the value function of the worker
    cost_model: instance of :class:`.Model`
        the shared model with the cost application
    algorithm: instance of :class: `theano.function`
        gradient of the cost function of the model
    resized_width: int
        the width of the images that will be used by the agent
    resized_height: int
        the height of the images that will be used by the agent
    agent_history_length: int
        number of past frames that will be used by the agent
    checkpoint_interval: int
       number of steps between checkpoint saves
    shared_model : instance of :class:`.Common_Model_Wrapper`
      the shared model
    num_reps_eval: int
      number of repetitions of the evalution FIXME: not used
    gamma_rate: float
      discount rate when boostrapping the accumulated reward
    sample_argmax: bool
      True if the action of max prob should be chosen or False to choose
        a multinomial sampling strategy

    """

    def __init__(self, rng, num, env, batch_size, policy_network,
                 value_network, cost_function,
                 cost_model, algorithm, resized_width, resized_height,
                 agent_history_length, checkpoint_interval, shared_model,
                 num_reps_eval, gamma_rate=0.99,
                 sample_argmax=True, extracost_function=None):
        super(A3C_Agent, self).__init__()

        self.rng = rng
        self.num = num
        self.env = env
        self.batch_size = batch_size
        self.policy_network = policy_network
        self.value_network = value_network
        self.cost_function = cost_function
        self.cost_model = cost_model
        self.optim_algorithm = algorithm
        self.optim_algorithm.initialize()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.checkpoint_interval = checkpoint_interval
        self.shared_model = shared_model
        self.num_reps_eval = num_reps_eval
        self.extracost_function = extracost_function
        self.gamma_rate = gamma_rate

        if sample_argmax:
            self._sample_function = sample_argmax_action
        else:
            self._sample_function = sample_policy_action

    def one_step_acc(self, init_params, algorithm, model, batch):
        """ Accumulate gradients in a minibatch step by step

        Note: this function is not actually used

        Parameters:
        init_params: OrderedDict
          parameters when starting a batch
        algorithm: instance of :class: `theano.function`
          function to obtain the gradient in a minibatch
        model: instance of :class: `~blocks.Model`
           cost model where parameters are extracted
        batch: int

        """
        # Restore initial state
        new_update = OrderedDict()

        for i in range(len(batch['input_reward'])):
            new_batch = OrderedDict()
            new_batch['input_image'] = [batch['input_image'][i]]
            new_batch['input_actions'] = [batch['input_actions'][i]]
            new_batch['input_reward'] = [batch['input_reward'][i]]
            model.set_parameter_values(init_params)
            algorithm.process_batch(new_batch)
            end_params = extract_params_from_model(model)

            for kk in init_params:
                if kk in end_params:
                    if kk in new_update:
                        new_update[kk] += (init_params[kk] - end_params[kk])
                    else:
                        new_update[kk] = (init_params[kk] - end_params[kk])
            else:
                logger.error("{} was not part of the update".format(kk))

        model.set_parameter_values(init_params)

        return new_update

    def run(self):
        """ Runs the Worker """
        logger.info("Thread {} running!".format(self.num))
        self.pick_one_thread_data()

    def pick_one_thread_data(self):
        """ Executes the iterations till training is over """

        # Wrap env with AtariEnvironment helper class
        env = AtariEnvironment(
            gym_env=self.env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length)

        # Add different starting time for each agent
        time.sleep(5*self.num)

        # Set up per-episode counters
        ep_reward = 0
        ep_t = 0

        probs_summary_t = 0

        # Picking initial state
        s_t = env.get_initial_state()
        terminal = False

        while (self.shared_model.common_model.curr_steps <
               self.shared_model.common_model.max_steps):
            s_batch = []
            past_rewards = []
            a_batch = []
            t = 0
            t_start = t

            # Update the model with the shared_model params (synchro)
            self.shared_model.synchronize_model(self.cost_model)

            # Store the initial params of the model
            init_params = extract_params_from_model(self.cost_model)

            # Execute one iteration
            while not (terminal or ((t - t_start) == self.batch_size)):
                probs = self.policy_network([s_t])[0][0]
                action_index = self._sample_function(
                    len(env.gym_actions), probs, self.rng)
                a_t = np.zeros([len(env.gym_actions)])
                a_t[action_index] = 1

                s_batch.append(s_t)
                a_batch.append(a_t)

                # Execute the action and obtain the reward
                s_t1, r_t, terminal, info = env.step(action_index)
                ep_reward += r_t

                r_t = np.clip(r_t, -1, 1)
                past_rewards.append(r_t)

                t += 1
                self.shared_model.common_model.curr_steps += 1
                ep_t += 1
                probs_summary_t += 1
                s_t = s_t1

            if terminal:
                R_t = 0
            else:
                # Picking last state
                R_t = self.value_network([s_t])[0][0]

            # Obtaining the reward at each epoch
            R_batch = np.zeros(t)

            for i in reversed(range(t_start, t)):
                R_t = past_rewards[i] + self.gamma_rate * R_t
                R_batch[i] = R_t

            # Minimize gradient
            batch = OrderedDict()
            batch['input_image'] = s_batch
            batch['input_actions'] = np.array(a_batch, dtype="int32")
            batch['input_reward'] = np.array(R_batch, dtype="float32")

            # Show stats each 1000 iterations
            if (self.shared_model.common_model.curr_it % 1000 == 0):
                # Show progress
                logger.info("Reward in batch {}".format(batch['input_reward']))

                logger.info("Current IT {} Steps {}".format(
                    self.shared_model.common_model.curr_it,
                    self.shared_model.common_model.curr_steps))
                logger.info("Cost network {}".format(self.cost_function(
                    batch['input_image'],
                    batch['input_reward'],
                    batch['input_actions'])[0]))

                logger.info("PROBS {}".format(
                    self.policy_network([s_t])[0][0]))
                logger.info("VALUES {}".format(self.value_network([s_t])[0]))

            # Picking last value for stats
            last_value = self.value_network([s_t])[0]

            # Perform basic gradient descent
            self.optim_algorithm.process_batch(batch)
            # update common parameters
            end_params = extract_params_from_model(self.cost_model)

            self.shared_model.update_cum_gradients(
                init_params,
                end_params,
                batch_size=t-t_start,
                stats_flag=((self.num == 0) and
                            ((self.shared_model.common_model.curr_it %
                              100 == 0)) or (
                         self.shared_model.common_model.curr_it == 1)))

            self.shared_model.common_model.curr_it += 1
            if ((self.shared_model.common_model.curr_it %
                 self.checkpoint_interval) == 0):
                # FIXME: Perform evaluation: we are going to scape this as
                # gym env fails to close some env after a test
                # self.shared_model.perform_evaluation( self.num_reps_eval,
                #                     self.shared_model.common_model.game,
                #              self.shared_model.common_model.curr_steps)

                # save progress
                self.shared_model.save_model("{}_{}.tar".format(
                    self.shared_model.common_model.game,
                    self.shared_model.common_model.curr_steps))

            # Check if terminal
            if terminal:
                logger.info("Episode Reward\t{}\tEpisode Length\t{}\tValue " +
                            "terminal state\t{}".format(
                                ep_reward, ep_t, last_value))
                # TODO: collect stats
                s_t = env.get_initial_state()
                terminal = False
                ep_reward = 0
                ep_t = 0


class MultiA3CTrainStream(object):
    """ Run an A3C Agent to collect training data and update an A3C model

    Parameters
    ----------
    rng : numpy.random
    epochs : int
      number of epochs FIXME: not used
    max_steps: int
        maximum number of steps the agent will play during training
    batch_size: int
      maximum size of a training batch
    game : str
        gym name of the game
    num_threads : int
      number of workers (training threads)
    resized_width: int
        the width of the images that will be used by the agent
    resized_height: int
        the height of the images that will be used by the agent
    agent_history_length: int
        number of past frames that will be used by the agent
    checkpoint_interval : int
        batch intervals at which a checkpoint is made
    training_flag: bool
     True if training should be performed
    render_flag: bool
        True to render the screen whilst evaluating the model
    results_file: str
        prefix path for storing the results
    num_reps_eval: int
      number of repetitions of the evalution FIXME: not used
    learning_rate: float
        learning rate during training
    gamma_rate: float
      discount rate when boostrapping the accumulated reward
    gradient_clipping: float
       a float number to clip gradients to this value
       (Default is None)
    model_file: str
       path of a previously stored  model to load

    """

    def __init__(self, rng, epochs, max_steps, batch_size, game, num_threads,
                 resized_width=84, resized_height=84, agent_history_length=4,
                 checkpoint_interval=5000, training_flag=True,
                 render_flag=False, results_file=False,
                 sample_argmax=True, num_steps_eval=1000,
                 num_reps_eval=10, learning_rate=0.00025,
                 gamma_rate=0.99,
                 gradient_clipping=None, model_file=None,
                 a3c_lstm=False,
                 lstm_output_units=256):
        """ Initialize stuff

        """

        self.rng = rng
        self.epochs = epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.game = game
        self.num_threads = num_threads
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.agent_history_length = agent_history_length
        self.checkpoint_interval = checkpoint_interval
        self.training_flag = training_flag
        self.num_steps_eval = num_steps_eval
        self.num_reps_eval = num_reps_eval
        self.render_flag = render_flag
        self.results_file = results_file
        self.sample_argmax = sample_argmax
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping
        self.model_file = model_file
        self.gamma_rate = gamma_rate
        self.a3c_lstm = a3c_lstm
        self.lstm_output_units = lstm_output_units

        # Build shared envs
        # TODO: check
        self.env = gym.make(self.game)
        self.validation_env = AtariEnvironment(
            gym_env=self.env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length)

    def training(self):
        """ Perform the training steps """

        # Create the envs of the threaded workers
        envs = [gym.make(self.game) for i in range(self.num_threads)]

        # Build the networks (one for each environment)
        a3c_networks = [A3C.build_a3c_network(
            image_size=(self.resized_width,
                        self.resized_height),
            num_channels=self.agent_history_length,
            num_actions=len(self.validation_env.gym_actions),
            lr=self.learning_rate,
            clip_c=self.gradient_clipping) for
                      thread_id in range(self.num_threads)]

        logger.info("Building the shared networks")
        a3c_global = A3C.build_a3c_network(
            image_size=(self.resized_width, self.resized_height),
            num_channels=self.agent_history_length,
            num_actions=len(self.validation_env.gym_actions),
            lr=self.learning_rate,
            clip_c=self.gradient_clipping,
            async_update=True)

        logger.info("Building the shared worker")
        # Building the extra environment for evaluation
        a3c_global_costmodel = Common_Model_Wrapper(
            rng=self.rng,
            game=self.game,
            model=a3c_global[0],
            algorithm=a3c_global[3],
            policy_network=a3c_global[1],
            value_network=a3c_global[2],
            monitor_env=self.env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length,
            max_steps=self.max_steps,
            num_steps_eval=self.num_steps_eval,
            sample_argmax=self.sample_argmax,
            results_file=self.results_file,
            render_flag=self.render_flag)

        # Start num concurrent threads
        thread_list = [A3C_Agent(
            rng=self.rng,
            num=thread_id,
            env=envs[thread_id],
            batch_size=self.batch_size,
            policy_network=a3c_networks[thread_id][1],
            value_network=a3c_networks[thread_id][2],
            cost_model=a3c_networks[thread_id][0],
            algorithm=a3c_networks[thread_id][3],
            cost_function=a3c_networks[thread_id][4],
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length,
            checkpoint_interval=self.checkpoint_interval,
            shared_model=a3c_global_costmodel,
            num_reps_eval=self.num_reps_eval,
            gamma_rate=self.gamma_rate,
            sample_argmax=self.sample_argmax,
            extracost_function=a3c_networks[thread_id][5])
                        for thread_id in range(self.num_threads)]

        for t in thread_list:
            t.start()

        # TODO: summary information here

        for t in thread_list:
            t.join()

    def do_test(self):
        """ Execute a gym evaluation """

        logger.info("Building the shared networks")
        a3c_global = A3C.build_a3c_network(
            image_size=(self.resized_width, self.resized_height),
            num_channels=self.agent_history_length,
            num_actions=len(self.validation_env.gym_actions),
            lr=self.learning_rate,
            clip_c=self.gradient_clipping,
            async_update=True)

        logger.info("Building the shared worker")
        # Building the extra environment for evaluation
        a3c_global_costmodel = Common_Model_Wrapper(
            rng=self.rng,
            game=self.game,
            model=a3c_global[0],
            algorithm=a3c_global[3],
            policy_network=a3c_global[1],
            value_network=a3c_global[2],
            monitor_env=self.env,
            resized_width=self.resized_width,
            resized_height=self.resized_height,
            agent_history_length=self.agent_history_length,
            max_steps=self.max_steps,
            num_steps_eval=self.num_steps_eval,
            sample_argmax=self.sample_argmax,
            results_file=self.results_file,
            render_flag=self.render_flag)

        if self.model_file is not None:
            print "LOADING PREVIOUS MODEL"
            with open(self.model_file, 'rb') as src:
                parameters = serialization.load_parameters(src)
                a3c_global_costmodel.common_model.model.set_parameter_values(
                    parameters)

        # Use the current time to build the experiment name
        exp_name = "{}_{}".format(self.game,
                                  time.strftime("%Y%m%d%H%M%S"))

        # Perform an evaluation
        a3c_global_costmodel.perform_evaluation(
            self.num_reps_eval, exp_name,
            0, do_gym_eval=True)

    def execute(self):
        """ Perform training/evaluation of the model """
        if self.training_flag:
            self.training()
        else:
            self.do_test()
