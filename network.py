import theano
import logging
from theano import tensor as T

from blocks.bricks.base import application
from blocks.initialization import (Constant, Uniform)

import numpy as np
from toolz.itertoolz import interleave
from collections import Counter
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener)
from blocks.bricks import (Initializable, Rectifier, FeedforwardSequence,
                           MLP, Activation, Softmax, Linear)

from blocks.model import Model
from blocks.graph import ComputationGraph

from blocks.algorithms import (Adam, GradientDescent, CompositeRule,
                               StepClipping, Scale)
from newblocks import (AsyncUpdate, AsyncRMSProp)

from blocks.bricks.recurrent import LSTM

logger = logging.getLogger(__name__)


class SharedA3CConvNet(FeedforwardSequence, Initializable):
    """ Implements the Shared Layers of the Actor-Critic

    Parameters
    ----------
    conv_activations : list of `blocks.bricks.base.brick`
        activation functions after every convolutional layers
    num_channels : int
      input channels in the first convolution layer. It is the number
      of historic frames used as the input state of the agent.
    image_shape : list of int
      width and height shape of the resized image
    filter_sizes: list of int  # FIXME: change the name
      num of filters at each convolutional layer
    feature_maps : list of [int, int]
       size of the filters (width, height) at each convolutional layer
    pooling sizes: list of [int,int]  # FIXME: not used
       size of the pooling layer. One element per convolutional layer
    mlp_hiddens: list of int
      size of the output layer of the hidden layers. One element per
      hidden layer.
    mlp_activations: list of `blocks.bricks.base.brick`
      activation functions at each hidden layer. One element per layer
    conv_step: list of (int, int)
      typically called stride
    border_mode : str
      full or valid are accepted by Blocks. Full will be usually employed.

    """

    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes, mlp_hiddens,
                 mlp_activations, conv_step=None, border_mode='valid',
                 **kwargs):

        if conv_step is None:
            self.conv_step = [(1, 1) for i in range(len(conv_activations))]
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.border_mode = border_mode
        self.top_mlp_activations = mlp_activations

        conv_parameters = zip(filter_sizes, feature_maps)

        # Build convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step[i],
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations]))

        # Build the sequence of conv layers
        self.conv_sequence = ConvolutionalSequence(self.layers, num_channels,
                                                   image_size=image_shape)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.top_mlp_dims = mlp_hiddens

        # Flatten the output so it can be used by DenseLayers
        self.flattener = Flattener()
        self.top_mlp = MLP(self.top_mlp_activations, self.top_mlp_dims)

        application_methods = [self.conv_sequence.apply,
                               self.flattener.apply,
                               self.top_mlp.apply]

        # FIXME this line was commented
        # self.children = [self.conv_sequence, self.flattener, self.top_mlp]

        super(SharedA3CConvNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        self.conv_sequence._push_allocation_config()
        conv_out_dim = self.conv_sequence.get_dim('output')

        print "Input to MLP hidden layer ", [np.prod(conv_out_dim)]
        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims

        print "Dimensions of hidden layer", self.top_mlp.dims


class PolicyAndValueA3C(Initializable):
    """
    Parameters
    ----------
    conv_activations : list of `blocks.bricks.base.brick`
        activation functions after every convolutional layers
    num_channels : int
      input channels in the first convolution layer. It is the number
      of historic frames used as the input state of the agent.
    image_shape : list of int
      width and height shape of the resized image
    filter_sizes: list of int  # FIXME: change the name
      num of filters at each convolutional layer
    feature_maps : list of [int, int]
       size of the filters (width, height) at each convolutional layer
    pooling sizes: list of [int,int]  # FIXME: not used
       size of the pooling layer. One element per convolutional layer
    mlp_hiddens: list of int
      size of the output layer of the hidden layers. One element per
      hidden layer.
    number_actions: int
      number of actions of the Actor (output of the policy network)
    mlp_activations: list of `blocks.bricks.base.brick`
      activation functions at each hidden layer. One element per layer
    activation_policy: instance of :class: `blocks.bricks.base.brick`
       activation at the policy layer. Tipically a Softmax because we
       want the probabilities of each action
    activation_value: instance of :class: `blocks.bricks.base.brick`
       the original function is a Linear one which is the default in Blocks.
       So None is the default.
    conv_step: list of (int, int)
      typically called stride
    border_mode : str
      full or valid are accepted by Blocks. Full will be usually employed.
    beta: float
      entropy error modulator. Default is 0.01

    """

    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes, mlp_hiddens,
                 number_actions, mlp_activations,
                 activation_policy=Softmax(), activation_value=None,
                 conv_step=None, border_mode='valid', beta=1e-2, **kwargs):

        self.activation_policy = activation_policy
        self.activation_value = activation_value
        self.beta = beta

        self.shared_a3c = SharedA3CConvNet(conv_activations=conv_activations,
                                           num_channels=num_channels,
                                           image_shape=image_shape,
                                           filter_sizes=filter_sizes,
                                           feature_maps=feature_maps,
                                           pooling_sizes=pooling_sizes,
                                           mlp_hiddens=mlp_hiddens,
                                           mlp_activations=mlp_activations,
                                           conv_step=conv_step,
                                           border_mode=border_mode, **kwargs)

        # We build now the policy/value separated networks
        print("Dimenson of the last shared layer {}".format(
            self.shared_a3c.top_mlp_dims[-1]))

        # Policy has one dimension per each action
        self.policy = MLP([activation_policy], [
                          self.shared_a3c.top_mlp_dims[-1]] +
                          [number_actions], name="mlp_policy")

        # The critic has one dimension in the output layer
        self.value = MLP([activation_value], [
                         self.shared_a3c.top_mlp_dims[-1]] + [1],
                         name="mlp_value")

        super(PolicyAndValueA3C, self).__init__(**kwargs)
        #self.children = [ self.shared_a3c, self.fork]
        self.children = [self.shared_a3c, self.policy, self.value]

    @application(inputs=['input_image'], outputs=['output_policy'])
    def apply_policy(self, input_image):
        output_policy = self.policy.apply(self.shared_a3c.apply(input_image))
        return output_policy

    @application(inputs=['input_image'], outputs=['output_value'])
    def apply_value(self, input_image):
        output_value = self.value.apply(self.shared_a3c.apply(input_image))
        return output_value

    # FIXME: remove this function
    @application(inputs=['input_image', 'input_actions', 'input_reward'],
                 outputs=['total_error', 'p_loss', 'v_loss', 'entropy',
                          'log_prob', 'advantage', 'v_value',
                          'sum_p_value'])
    def extra_cost(self, input_image, input_actions, input_reward):

        p_value = self.policy.apply(self.shared_a3c.apply(input_image))
        log_prob = T.log(T.sum(p_value * input_actions, axis=1, keepdims=True))
        v_value = self.value.apply(self.shared_a3c.apply(input_image))
        advantage = (input_reward[:, None] - v_value)
        p_loss = -1 * log_prob * advantage

        entropy = -T.sum(p_value * T.log(p_value), axis=1, keepdims=True)
        p_loss = p_loss - self.beta * entropy  # add entropy penalty
        v_loss = 0.5 * T.sqr(input_reward[:, None] - v_value)

        total_error = T.mean(p_loss + (0.5 * v_loss.reshape(p_loss.shape)))

        return (total_error, p_loss, v_loss, entropy, log_prob, advantage,
                v_value, T.sum(p_value, axis=1))

    @application(inputs=['input_image', 'input_actions', 'input_reward'],
                 outputs=['total_error'])
    def cost(self, input_image, input_actions, input_reward):

        p_value = (self.policy.apply(self.shared_a3c.apply(input_image)))
        log_prob = T.log(T.sum((p_value) * input_actions,
                               axis=1, keepdims=True))
        v_value = self.value.apply(self.shared_a3c.apply(input_image))
        p_loss = -log_prob * theano.gradient.disconnected_grad(
            input_reward[:, None] - v_value)

        entropy = -T.sum(p_value * T.log(p_value), axis=1,
                         keepdims=True)
        # encourage action diversity by substracting entropy
        p_loss = p_loss - self.beta * entropy
        v_loss = T.sqr(input_reward[:, None] - v_value)

        total_error = T.mean(p_loss + (0.5 * v_loss))
        return total_error


class PolicyAndValueA3CLSTM(Initializable):
    """
    Parameters
    ----------
    conv_activations : list of `blocks.bricks.base.brick`
        activation functions after every convolutional layers
    num_channels : int
      input channels in the first convolution layer. It is the number
      of historic frames used as the input state of the agent.
    image_shape : list of int
      width and height shape of the resized image
    filter_sizes: list of int  # FIXME: change the name
      num of filters at each convolutional layer
    feature_maps : list of [int, int]
       size of the filters (width, height) at each convolutional layer
    pooling sizes: list of [int,int]  # FIXME: not used
       size of the pooling layer. One element per convolutional layer
    mlp_hiddens: list of int
      size of the output layer of the hidden layers. One element per
      hidden layer.
    number_actions: int
      number of actions of the Actor (output of the policy network)
    mlp_activations: list of `blocks.bricks.base.brick`
      activation functions at each hidden layer. One element per layer
    activation_policy: instance of :class: `blocks.bricks.base.brick`
       activation at the policy layer. Tipically a Softmax because we
       want the probabilities of each action
    activation_value: instance of :class: `blocks.bricks.base.brick`
       the original function is a Linear one which is the default in Blocks.
       So None is the default.
    conv_step: list of (int, int)
      typically called stride
    border_mode : str
      full or valid are accepted by Blocks. Full will be usually employed.
    beta: float
      entropy error modulator. Default is 0.01
    lstm_output_units: int
      number of LSTM output units

    """

    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes, mlp_hiddens,
                 number_actions, mlp_activations,
                 activation_policy=Softmax(), activation_value=None,
                 conv_step=None, border_mode='valid', beta=1e-2,
                 lstm_output_units=None, **kwargs):

        self.activation_policy = activation_policy
        self.activation_value = activation_value
        self.beta = beta
        self.lstm_output_units = lstm_output_units

        self.shared_a3c = SharedA3CConvNet(conv_activations=conv_activations,
                                           num_channels=num_channels,
                                           image_shape=image_shape,
                                           filter_sizes=filter_sizes,
                                           feature_maps=feature_maps,
                                           pooling_sizes=pooling_sizes,
                                           mlp_hiddens=mlp_hiddens,
                                           mlp_activations=mlp_activations,
                                           conv_step=conv_step,
                                           border_mode=border_mode, **kwargs)

        # We build now the policy/value separated networks
        print("Dimenson of the last shared layer {}".format(
            self.shared_a3c.top_mlp_dims[-1]))

        # LSTM block

        # Preparation to LSTM
        print "LSTM UNITS ", self.lstm_output_units
        self.linear_to_lstm = Linear(self.shared_a3c.top_mlp_dims[-1],
                                     self.lstm_output_units * 4,
                                     name='linear_to_lstm')
        self.lstm_block = LSTM(lstm_output_units, name='lstm')
        # activation=Rectifier())

        # Policy has one dimension per each action
        self.policy = MLP([activation_policy], [
                          lstm_output_units] +
                          [number_actions], name="mlp_policy")

        # The critic has one dimension in the output layer
        self.value = MLP([activation_value], [
                         lstm_output_units] + [1],
                         name="mlp_value")

        super(PolicyAndValueA3CLSTM, self).__init__(**kwargs)

        self.children = [self.shared_a3c, self.linear_to_lstm,
                         self.lstm_block,
                         self.policy, self.value]

    @application(inputs=['input_image', 'states', 'cells'],
                 outputs=['output_policy', 'states', 'cells'])
    def apply_policy(self, input_image, states, cells):

        h, c = self.lstm_block.apply(inputs=self.linear_to_lstm.apply(
            self.shared_a3c.apply(input_image)),
                                     states=states, cells=cells)

        h = h.sum(axis=1)
        c = c.sum(axis=1)

        output_policy = self.policy.apply(h)
        return output_policy, h, c

    @application(inputs=['input_image', 'states', 'cells'],
                 outputs=['output_value'])
    def apply_value(self, input_image, states, cells):
        h, c = self.lstm_block.apply(inputs=self.linear_to_lstm.apply(
            self.shared_a3c.apply(input_image)),
                                     states=states, cells=cells)

        h = h.sum(axis=1)
        c = c.sum(axis=1)

        output_value = self.value.apply(h)
        return output_value

    @application(inputs=['input_image', 'input_actions', 'input_reward',
                         'states', 'cells'],
                 outputs=['total_error'])
    def cost(self, input_image, input_actions, input_reward, states, cells):

        h, c = self.lstm_block.apply(inputs=self.linear_to_lstm.apply(
            self.shared_a3c.apply(input_image)),
                                     states=states, cells=cells)
        h = h.sum(axis=1)
        c = c.sum(axis=1)

        p_value = self.policy.apply(h)
        log_prob = T.log(T.sum((p_value) * input_actions,
                               axis=1, keepdims=True))
        v_value = self.value.apply(h)
        p_loss = -log_prob * theano.gradient.disconnected_grad(
            input_reward[:, None] - v_value)

        entropy = -T.sum(p_value * T.log(p_value), axis=1,
                         keepdims=True)
        # encourage action diversity by substracting entropy
        p_loss = p_loss - self.beta * entropy
        v_loss = T.sqr(input_reward[:, None] - v_value)

        total_error = T.mean(p_loss + (0.5 * v_loss))
        return total_error


def build_a3c_network(feature_maps=[16, 32],
                      conv_sizes=[8, 4],
                      pool_sizes=[4, 2],
                      # FIXME: used image_shape elsewhere
                      image_size=(80, 80),
                      step_size=[4, 2],
                      num_channels=10,
                      mlp_hiddens=[256],
                      num_actions=10,
                      lr=0.00025,
                      clip_c=0.8,
                      border_mode='full',
                      async_update=False):
    """ Builds the agent networks/functions

    Parameters:
    -----------
    feature_maps : list of [int, int]
       size of the filters (width, height) at each convolutional layer
    conv_sizes: list of int  # FIXME: change the name
      num of filters at each convolutional layer
    pooling sizes: list of int  # FIXME: not used
       size of the pooling layer. One element per convolutional layer
    image_size : list of int
      width and height shape of the resized image
    step_size: list of int
      typically called stride
    num_channels : int
      input channels in the first convolution layer. It is the number
      of historic frames used as the input state of the agent.
    mlp_hiddens: list of int
      size of the output layer of the hidden layers. One element per
      hidden layer.
    num_actions: int
      number of actions of the Actor (output of the policy network)
    lr : float

      learning rate of async rmsprop
    clip_c : float
      > 0 if gradient should be clipped. FIXME: actually not used
    border_mode : str
      full or valid are accepted by Blocks. Full will be usually employed.
    async_update: bool
      true if the network to be created is the shared worker or False if
      it is just a worker.

    """

    # Activation functions
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens]
    conv_subsample = [[step, step] for step in step_size]

    policy_and_value_net = PolicyAndValueA3C(
        conv_activations,
        num_channels,
        image_size,
        filter_sizes=zip(conv_sizes, conv_sizes),
        feature_maps=feature_maps,
        pooling_sizes=zip(pool_sizes, pool_sizes),
        mlp_hiddens=mlp_hiddens,
        number_actions=num_actions,
        mlp_activations=mlp_activations,
        conv_step=conv_subsample,
        border_mode='full',
        weights_init=Uniform(width=.2),
        biases_init=Constant(.0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    policy_and_value_net.shared_a3c.push_initialization_config()
    policy_and_value_net.push_initialization_config()

    # Xavier initialization
    for i in range(len(policy_and_value_net.shared_a3c.layers)):
        if i == 0:
            policy_and_value_net.shared_a3c.layers[i].weights_init = Uniform(
                std=1.0/np.sqrt((image_size[0] *
                                 image_size[1] *
                                 num_channels)))
        else:
            policy_and_value_net.shared_a3c.layers[i].weights_init = Uniform(
                std=1.0/np.sqrt((conv_sizes[(i-1)/2] *
                                 conv_sizes[(i-1)/2] *
                                 feature_maps[(i-1)/2])))

        policy_and_value_net.shared_a3c.layers[i].bias_init = Constant(.1)

    for i in range(len(policy_and_value_net.shared_a3c.
                       top_mlp.linear_transformations)):
        policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
            i].weights_init = Uniform(std=1.0/np.sqrt((conv_sizes[-1] *
                                                       conv_sizes[-1] *
                                                       feature_maps[-1])))
        policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
            i].bias_init = Constant(.0)

    policy_and_value_net.policy.weights_init = Uniform(
        std=1.0/np.sqrt(mlp_hiddens[-1]))
    policy_and_value_net.value.weights_init = Uniform(
        std=1.0/np.sqrt(mlp_hiddens[-1]))
    policy_and_value_net.shared_a3c.initialize()
    policy_and_value_net.initialize()
    logging.info("Input dim: {} {} {}".format(
        *policy_and_value_net.shared_a3c.children[0].get_dim('input_')))
    for i, layer in enumerate(policy_and_value_net.shared_a3c.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    th_input_image = T.tensor4('input_image')
    th_reward = T.fvector('input_reward')
    th_actions = T.imatrix('input_actions')

    policy_network = policy_and_value_net.apply_policy(th_input_image)
    value_network = policy_and_value_net.apply_value(th_input_image)
    cost_network = policy_and_value_net.cost(th_input_image, th_actions,
                                             th_reward)
    # FIXME: added for debug, remove
    extracost_network = policy_and_value_net.extra_cost(th_input_image,
                                                        th_actions,
                                                        th_reward)  # DEBUG

    cg_policy = ComputationGraph(policy_network)
    cg_value = ComputationGraph(value_network)

    # Perform some optimization step
    cg = ComputationGraph(cost_network)

    # FIXME: Remove
    cg_extra = ComputationGraph(extracost_network)  # DEBUG

    # Print shapes of network parameters
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    cost_model = Model(cost_network)
    value_model = Model(value_network)

    if not async_update:
        # A threaded worker: steep gradient descent
        # A trick was done here to reuse existent bricks. The system performed
        # steepest descent to aggregate the gradients. However, the gradients
        # are averaged in a minibatch (instead of being just added). Therefore,
        # the agent is going to perform the following operations in each
        # minibatch:
        # 1) steepes descent with learning rate of 1 to only aggregate the
        # gradients.
        # 2) undo the update operation to obtain the avg. gradient :
        #    gradient = parameter_before_minibatch - parameter_after_minibatch
        # 3) Multiply the gradient by the length of the minibatch to obtain the
        #    exact gradient at each minibatch.
        algorithm = GradientDescent(
            cost=cost_network, parameters=cg.parameters,
            step_rule=Scale())
    else:
        # Async update for the shared worker
        # The other part of the trick. A custom optimization block was
        # developed
        # here to receive as inputs the acc. gradients at each worker
        algorithm = AsyncUpdate(parameters=cg.parameters,
                                inputs=cost_model.get_parameter_dict().keys(),
                                step_rule=AsyncRMSProp(learning_rate=lr,
                                                       # FIXME: put as
                                                       # parameter
                                                       decay_rate=0.99,
                                                       max_scaling=10))

    algorithm.initialize()

    f_cost = theano.function(inputs=cg.inputs, outputs=cg.outputs)
    f_policy = theano.function(inputs=cg_policy.inputs,
                               outputs=cg_policy.outputs)
    f_value = theano.function(inputs=cg_value.inputs, outputs=cg_value.outputs)

    # f_extracost = theano.function(inputs=cg_extra.inputs,
    #                               outputs=cg_extra.outputs)

    return cost_model, f_policy, f_value, algorithm, f_cost


def build_a3c_network_lstm(feature_maps=[16, 32],
                           conv_sizes=[8, 4],
                           pool_sizes=[4, 2],
                           # FIXME: used image_shape elsewhere
                           image_size=(80, 80),
                           step_size=[4, 2],
                           num_channels=10,
                           mlp_hiddens=[256],
                           lstm_output_units=256,
                           num_actions=10,
                           lr=0.00025,
                           clip_c=0.8,
                           border_mode='full',
                           async_update=False):
    """ Builds the agent networks/functions

    Parameters:
    -----------
    feature_maps : list of [int, int]
       size of the filters (width, height) at each convolutional layer
    conv_sizes: list of int  # FIXME: change the name
      num of filters at each convolutional layer
    pooling sizes: list of int  # FIXME: not used
       size of the pooling layer. One element per convolutional layer
    image_size : list of int
      width and height shape of the resized image
    step_size: list of int
      typically called stride
    num_channels : int
      input channels in the first convolution layer. It is the number
      of historic frames used as the input state of the agent.
    mlp_hiddens: list of int
      size of the output layer of the hidden layers. One element per
      hidden layer.
    lstm_output_units: int
      number of units in the lstm output
    num_actions: int
      number of actions of the Actor (output of the policy network)
    lr : float
      learning rate of async rmsprop
    clip_c : float
      > 0 if gradient should be clipped. FIXME: actually not used
    border_mode : str
      full or valid are accepted by Blocks. Full will be usually employed.
    async_update: bool
      true if the network to be created is the shared worker or False if
      it is just a worker.

    """

    # Activation functions
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens]
    conv_subsample = [[step, step] for step in step_size]

    policy_and_value_net = PolicyAndValueA3CLSTM(
        conv_activations,
        num_channels,
        image_size,
        filter_sizes=zip(conv_sizes, conv_sizes),
        feature_maps=feature_maps,
        pooling_sizes=zip(pool_sizes, pool_sizes),
        mlp_hiddens=mlp_hiddens,
        lstm_output_units=lstm_output_units,
        number_actions=num_actions,
        mlp_activations=mlp_activations,
        conv_step=conv_subsample,
        border_mode='full',
        weights_init=Uniform(width=.2),
        biases_init=Constant(.0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    policy_and_value_net.shared_a3c.push_initialization_config()
    policy_and_value_net.push_initialization_config()

    # Xavier initialization
    for i in range(len(policy_and_value_net.shared_a3c.layers)):
        if i == 0:
            policy_and_value_net.shared_a3c.layers[i].weights_init = Uniform(
                std=1.0/np.sqrt((image_size[0] *
                                 image_size[1] *
                                 num_channels)))
        else:
            policy_and_value_net.shared_a3c.layers[i].weights_init = Uniform(
                std=1.0/np.sqrt((conv_sizes[(i-1)/2] *
                                 conv_sizes[(i-1)/2] *
                                 feature_maps[(i-1)/2])))

        policy_and_value_net.shared_a3c.layers[i].bias_init = Constant(.1)

    for i in range(len(policy_and_value_net.shared_a3c.
                       top_mlp.linear_transformations)):
        policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
            i].weights_init = Uniform(std=1.0/np.sqrt((conv_sizes[-1] *
                                                       conv_sizes[-1] *
                                                       feature_maps[-1])))
        policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
            i].bias_init = Constant(.0)

    policy_and_value_net.linear_to_lstm.weights_init = Uniform(
        std=1.0/np.sqrt(mlp_hiddens[-1]))
    policy_and_value_net.linear_to_lstm.biases_init = Constant(.0)
    policy_and_value_net.linear_to_lstm.initialize()
    policy_and_value_net.lstm_block.weights_init = Uniform(
        std=1.0/np.sqrt(mlp_hiddens[-1]))
    policy_and_value_net.lstm_block.biases_init = Constant(.0)
    policy_and_value_net.lstm_block.initialize()

    policy_and_value_net.policy.weights_init = Uniform(
        std=1.0/np.sqrt(lstm_output_units))
    policy_and_value_net.value.weights_init = Uniform(
        std=1.0/np.sqrt(lstm_output_units))
    policy_and_value_net.shared_a3c.initialize()
    policy_and_value_net.initialize()
    logging.info("Input dim: {} {} {}".format(
        *policy_and_value_net.shared_a3c.children[0].get_dim('input_')))
    for i, layer in enumerate(policy_and_value_net.shared_a3c.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    th_input_image = T.tensor4('input_image')
    th_reward = T.fvector('input_reward')
    th_actions = T.imatrix('input_actions')
    th_states = T.matrix('states')
    th_cells = T.matrix('cells')

    policy_network = policy_and_value_net.apply_policy(th_input_image,
                                                       th_states,
                                                       th_cells)
    value_network = policy_and_value_net.apply_value(th_input_image,
                                                     th_states,
                                                     th_cells)
    cost_network = policy_and_value_net.cost(th_input_image, th_actions,
                                             th_reward, th_states,
                                             th_cells)

    cg_policy = ComputationGraph(policy_network)
    cg_value = ComputationGraph(value_network)

    print "POLICY INPUTS ", cg_policy.inputs
    print "VALUE INPUTS ", cg_value.inputs

    print "POLICY OUTPUTS ", cg_policy.outputs
    print "VALUE OUTPUTS ", cg_value.outputs

    # Perform some optimization step
    cg = ComputationGraph(cost_network)

    # Print shapes of network parameters
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    cost_model = Model(cost_network)
    value_model = Model(value_network)   # FIXME: delete

    if not async_update:
        # A threaded worker: steep gradient descent
        # A trick was done here to reuse existent bricks. The system performed
        # steepest descent to aggregate the gradients. However, the gradients
        # are averaged in a minibatch (instead of being just added). Therefore,
        # the agent is going to perform the following operations in each
        # minibatch:
        # 1) steepes descent with learning rate of 1 to only aggregate the
        # gradients.
        # 2) undo the update operation to obtain the avg. gradient :
        #    gradient = parameter_before_minibatch - parameter_after_minibatch
        # 3) Multiply the gradient by the length of the minibatch to obtain the
        #    exact gradient at each minibatch.
        algorithm = GradientDescent(
            cost=cost_network, parameters=cg.parameters,
            step_rule=Scale())
    else:
        # Async update for the shared worker
        # The other part of the trick. A custom optimization block was
        # developed
        # here to receive as inputs the acc. gradients at each worker
        algorithm = AsyncUpdate(parameters=cg.parameters,
                                inputs=cost_model.get_parameter_dict().keys(),
                                step_rule=AsyncRMSProp(learning_rate=lr,
                                                       # FIXME: put as
                                                       # parameter
                                                       decay_rate=0.99,
                                                       max_scaling=10))

    algorithm.initialize()

    print "COST_INPUTS ", cg.inputs

    f_cost = theano.function(inputs=cg.inputs, outputs=cg.outputs)
    f_policy = theano.function(inputs=cg_policy.inputs,
                               outputs=cg_policy.outputs)
    f_value = theano.function(inputs=cg_value.inputs, outputs=cg_value.outputs)

    return cost_model, f_policy, f_value, algorithm, f_cost


if __name__ == "__main__":
    """ A small code snippet to test the network """

    feature_maps = [32, 64]
    conv_sizes = [8, 4]
    pool_sizes = [4, 2]
    image_size = (80, 80)
    step_size = [4, 2]

    num_channels = 10
    mlp_hiddens = [500, 256]
    num_actions = 10

    # dropout and gradient clipping
    dropout = 0.2
    clip_c = 0.8

    # Initialize policy network
    # Shared A3C
    # Initialize value network
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens]
    conv_subsample = [[step, step] for step in step_size]

    policy_and_value_net = PolicyAndValueA3C(conv_activations, num_channels,
                                             image_size,
                                             filter_sizes=zip(
                                                 conv_sizes, conv_sizes),
                                             feature_maps=feature_maps,
                                             pooling_sizes=zip(
                                                 pool_sizes, pool_sizes),
                                             mlp_hiddens=mlp_hiddens,
                                             number_actions=num_actions,
                                             mlp_activations=mlp_activations,
                                             conv_step=conv_subsample,
                                             border_mode='full',
                                             weights_init=Uniform(width=.2),
                                             biases_init=Constant(0))

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    policy_and_value_net.shared_a3c.push_initialization_config()
    policy_and_value_net.push_initialization_config()

    policy_and_value_net.shared_a3c.layers[0].weights_init = Uniform(width=.2)
    policy_and_value_net.shared_a3c.layers[1].weights_init = Uniform(width=.09)

    policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
        0].weights_init = Uniform(width=.08)

    policy_and_value_net.policy.weights_init = Uniform(width=.15)
    policy_and_value_net.value.weights_init = Uniform(width=.15)
    policy_and_value_net.shared_a3c.initialize()
    policy_and_value_net.policy.initialize()
    policy_and_value_net.value.initialize()
    policy_and_value_net.initialize()
    logging.info("Input dim: {} {} {}".format(
        *policy_and_value_net.shared_a3c.children[0].get_dim('input_')))
    for i, layer in enumerate(policy_and_value_net.shared_a3c.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    x = T.tensor4('features')

    policy = policy_and_value_net.apply_policy(x)
    value = policy_and_value_net.apply_value(x)
    num_batches = 32
    random_data = np.array(np.random.randint(128,
                                             size=(num_batches, num_channels,
                                                   image_size[0],
                                                   image_size[1])),
                           dtype="float32")

    pol_result = policy.eval({x: random_data})
    val_result = value.eval({x: random_data})

    print "POLICY SHAPE ", np.shape(pol_result)
    print "VALUE SHAPE ", np.shape(val_result)

    th_reward = T.vector('ereward')
    th_actions = T.imatrix('actions')

    reward = np.array(np.random.rand((num_batches)), dtype="float32")

    actions = np.zeros((num_batches, num_actions), dtype="int32")
    for i in range(0, num_batches):
        index_action = np.random.randint(num_actions)
        actions[i, index_action] = 1

    cost_network = policy_and_value_net.cost(x, th_actions, th_reward)
    cost_results = cost_network.eval(
        {x: random_data, th_actions: actions, th_reward: reward})

    # Perform some optimization step
    cg = ComputationGraph(cost_network)

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost_network, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(clip_c),
                                 Adam()]))

    cost_model = Model(cost_network)
    logger.info("Cost Model ".format(cost_model.get_parameter_dict()))

    # Check A3C-LSTM network
    lstm_output_units = mlp_hiddens[-1]

    policy_and_value_net_lstm = PolicyAndValueA3CLSTM(
        conv_activations, num_channels,
        image_size,
        filter_sizes=zip(
            conv_sizes, conv_sizes),
        feature_maps=feature_maps,
        pooling_sizes=zip(
            pool_sizes, pool_sizes),
        mlp_hiddens=mlp_hiddens,
        number_actions=num_actions,
        mlp_activations=mlp_activations,
        conv_step=conv_subsample,
        border_mode='full',
        weights_init=Uniform(width=.2),
        biases_init=Constant(0),
        lstm_output_units=lstm_output_units)

    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    policy_and_value_net_lstm.shared_a3c.push_initialization_config()
    policy_and_value_net_lstm.push_initialization_config()

    policy_and_value_net_lstm.shared_a3c.layers[
        0].weights_init = Uniform(width=.2)
    policy_and_value_net_lstm.shared_a3c.layers[
        1].weights_init = Uniform(width=.09)

    policy_and_value_net_lstm.shared_a3c.top_mlp.linear_transformations[
        0].weights_init = Uniform(width=.08)

    policy_and_value_net_lstm.policy.weights_init = Uniform(width=.15)
    policy_and_value_net_lstm.value.weights_init = Uniform(width=.15)
    policy_and_value_net_lstm.shared_a3c.initialize()
    policy_and_value_net_lstm.policy.initialize()
    policy_and_value_net_lstm.value.initialize()
    policy_and_value_net_lstm.linear_to_lstm.weights_init = Uniform(width=.15)
    policy_and_value_net_lstm.linear_to_lstm.biases_init = Constant(.0)
    policy_and_value_net_lstm.linear_to_lstm.initialize()
    policy_and_value_net_lstm.lstm_block.initialize()
    policy_and_value_net_lstm.initialize()
    logging.info("Input dim: {} {} {}".format(
        *policy_and_value_net_lstm.shared_a3c.children[0].get_dim('input_')))
    for i, layer in enumerate(policy_and_value_net_lstm.shared_a3c.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))

    x = T.tensor4('features')
    th_states = T.matrix('states')
    th_cells = T.matrix('cells')

    policy = policy_and_value_net_lstm.apply_policy(
        x, th_states,
        th_cells)
    value = policy_and_value_net_lstm.apply_value(
        x, th_states,
        th_cells)
    num_batches = 32
    random_data = np.array(np.random.randint(128,
                                             size=(num_batches, num_channels,
                                                   image_size[0],
                                                   image_size[1])),
                           dtype="float32")

    random_states = np.array(np.random.rand(1, lstm_output_units),
                             dtype="float32")
    random_cells = np.array(np.random.rand(1, lstm_output_units),
                            dtype="float32")

    pol_result = policy[0].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})
    val_result = value.eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    h_result = policy[1].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    c_result = policy[2].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    print "POLICY SHAPE LSTM ", np.shape(pol_result)
    print "VALUE SHAPE LSTM ", np.shape(val_result)
    print "H SHAPE LSTM ", np.shape(h_result)
    print "C SHAPE LSTM ", np.shape(c_result)

    th_reward = T.vector('ereward')
    th_actions = T.imatrix('actions')

    reward = np.array(np.random.rand((num_batches)), dtype="float32")

    actions = np.zeros((num_batches, num_actions), dtype="int32")
    for i in range(0, num_batches):
        index_action = np.random.randint(num_actions)
        actions[i, index_action] = 1

    cost_network = policy_and_value_net_lstm.cost(x, th_actions, th_reward,
                                                  th_states, th_cells)
    cost_results = cost_network.eval(
        {x: random_data, th_actions: actions, th_reward: reward,
         th_states: random_states, th_cells: random_cells})

    # Perform some optimization step
    cg = ComputationGraph(cost_network)

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

    # Set up training algorithm
    logger.info("Initializing training algorithm")
    algorithm = GradientDescent(
        cost=cost_network, parameters=cg.parameters,
        step_rule=CompositeRule([StepClipping(clip_c),
                                 Adam()]))

    cost_model = Model(cost_network)
    logger.info("Cost Model ".format(cost_model.get_parameter_dict()))

    # Check differnent result with batch

    random_data = np.array(np.random.randint(128,
                                             size=(1, num_channels,
                                                   image_size[0],
                                                   image_size[1])),
                           dtype="float32")

    random_data = np.concatenate((random_data, random_data),
                                 axis=0)

    print "RANDOM_INPUT_", np.shape(random_data)
    random_states = np.array(np.random.rand(1, lstm_output_units),
                             dtype="float32")
    random_cells = np.array(np.random.rand(1, lstm_output_units),
                            dtype="float32")

    pol_result = policy[0].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    next_state = policy[1].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    next_cell = policy[2].eval(
        {x: random_data,
         th_states: random_states,
         th_cells: random_cells})

    print "POLRESULT ", pol_result

    print "NEXT_STATE {} SUM0 {} SUM1 {}".format(np.shape(next_state),
                                                 np.sum(next_state[0]),
                                                 np.sum(next_state[1]))
    print "NEXT_CELL {} SUM0 {} SUM1 {}".format(np.shape(next_cell),
                                                np.sum(next_cell[0]),
                                                np.sum(next_cell[1]))

    # Do the same step by step

    prev_state = random_states
    prev_cell = random_cells

    pol_result = policy[0].eval(
        {x: [random_data[0]],
         th_states: prev_state,
         th_cells: prev_cell})

    next_state = policy[1].eval(
        {x: [random_data[0]],
         th_states: prev_state,
         th_cells: prev_cell})

    next_cell = policy[2].eval(
        {x: [random_data[0]],
         th_states: prev_state,
         th_cells: prev_cell})

    print "NEXT_STATE {} SUM1 {}".format(np.shape(next_state),
                                         np.sum(next_state))

    print "NEXT_CELL {} SUM1 {}".format(np.shape(next_cell),
                                        np.sum(next_cell))
