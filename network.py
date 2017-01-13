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
                           MLP, Activation, Softmax)

from blocks.model import Model
from blocks.graph import ComputationGraph

from blocks.algorithms import (Adam, GradientDescent, CompositeRule,
                               StepClipping, Scale)
from newblocks import (AsyncUpdate, AsyncRMSProp)
from collections import OrderedDict

logger = logging.getLogger(__name__)


class SharedA3CConvNet(FeedforwardSequence, Initializable):
    """

    Parameters
    -------------

    #TODO

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
    Parameters:
    -------------

    TODO

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


def build_a3c_network(feature_maps=[16, 32],
                      conv_sizes=[8, 4],
                      pool_sizes=[4, 2],
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
    # TODO

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

    f_extracost = theano.function(inputs=cg_extra.inputs,
                                  outputs=cg_extra.outputs)

    return cost_model, f_policy, f_value, algorithm, f_cost, f_extracost


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
    th_actions = T.matrix('actions')

    reward = np.array(np.random.rand((num_batches)), dtype="float32")
    actions = np.arange(0, num_actions, dtype="int32")

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
