import theano
import logging
from theano import tensor as T

import blocks
from blocks.bricks.base import application
from blocks.bricks.conv import MaxPooling
from blocks.initialization import (Constant, Uniform,
                                   IsotropicGaussian)
from toolz import merge
import logging
import numpy as np
from toolz.itertoolz import interleave
from collections import Counter
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener)
from blocks.bricks import (Initializable, Rectifier, Logistic,
                           FeedforwardSequence,
                           MLP, Activation, Linear, Softmax)

from blocks.model import Model
from blocks.bricks.parallel import Fork
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from newblocks import (Adam, RMSProp, GradientDescent, CompositeRule,
                       StepClipping, AsyncUpdate, Scale, AsyncRMSProp)
from collections import OrderedDict

logger = logging.getLogger(__name__)


def update_cum_gradients(init_state, last_state, common_net):
    """ Update the shared model parameters
    init_state (OrderedDict)
    last_state (OrderedDict)
    common_net (Model)

    """
    new_update = OrderedDict()

    for kk in init_state:
        if kk in last_state:
            new_update[kk] = last_state[kk] - init_state[kk]

    # FIXME perform the lock here

    # Update joint parameters then
    for kk in common_net.get_parameter_dict():
        if kk in new_update:
            print "UPDATE {} WITH MEAN {}".format(kk,
                                                  np.mean(new_update[kk]))

            common_net.get_parameter_dict()[kk].set_value(
                common_net.get_parameter_dict()[kk].get_value() +
                new_update[kk])

    # FIXME: Release the lock here


def extract_params_from_model(common_net):
    params = OrderedDict()

    for kk in common_net.get_parameter_dict():
        params[kk] = common_net.get_parameter_dict()[kk].get_value()

    return params


def read_common_parameters(agent_model, common_model):

    # A one liner here, just passing the parameters from model to model
    agent_model.set_parameter_values(common_model.get_parameter_values())


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

        # Construct convolutional layers with corresponding parameters
        self.layers = list(interleave([
            (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step[i],
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
             for i, (filter_size, num_filter)
             in enumerate(conv_parameters)),
            conv_activations]))

        # (MaxPooling(size, name='pool_{}'.format(i))
        # for i, size in enumerate(pooling_sizes))]))
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

        print "MY FLUXO ", [np.prod(conv_out_dim)]
        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [np.prod(conv_out_dim)] + self.top_mlp_dims

        print "CHNUF", self.top_mlp.dims


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

        # We build now the policy
        print "CHUF0 ", self.shared_a3c.top_mlp_dims[-1]

        self.policy = MLP([activation_policy], [
                          self.shared_a3c.top_mlp_dims[-1]] +
                          [number_actions], name="mlp_policy")
        self.value = MLP([activation_value], [
                         self.shared_a3c.top_mlp_dims[-1]] + [1],
                         name="mlp_value")

        super(PolicyAndValueA3C, self).__init__(**kwargs)
        #self.children = [ self.shared_a3c, self.fork]
        self.children = [self.shared_a3c, self.policy, self.value]

    @application(inputs=['input_image'], outputs=['output_policy'])
    def apply_policy(self, input_image):
        #output_policy = self.fork.apply(self.shared_a3c.apply(input_image))[0]
        output_policy = self.policy.apply(self.shared_a3c.apply(input_image))
        return output_policy

    @application(inputs=['input_image'], outputs=['output_value'])
    def apply_value(self, input_image):
        #output_value = self.fork.apply(self.shared_a3c.apply(input_image))[1]
        output_value = self.value.apply(self.shared_a3c.apply(input_image))
        return output_value

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
        p_loss = p_loss - self.beta * entropy  # encourage diversity
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
                      dropout=0.2,  # FIXME: not used: delete
                      lr=0.00025,
                      clip_c=0.8,
                      border_mode='full',
                      async_update=False):

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

    print "LEN ", len(policy_and_value_net.shared_a3c.layers)
    
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

    print "POLICY SPACE ", dir(policy_and_value_net.policy)
    print "VALUE SPACE ", dir(policy_and_value_net.value)

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
    extracost_network = policy_and_value_net.extra_cost(th_input_image,
                                                        th_actions,
                                                        th_reward)  # DEBUG

    cg_policy = ComputationGraph(policy_network)
    cg_value = ComputationGraph(value_network)

    # Perform some optimization step
    cg = ComputationGraph(cost_network)
    cg_extra = ComputationGraph(extracost_network)  # DEBUG

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    logger.info("Parameter shapes: ")
    for shape, count in Counter(shapes).most_common():
        logger.info('    {:15}: {}'.format(shape, count))
        logger.info("Total number of parameters: {}".format(len(shapes)))

    # Set up training algorithm
    logger.info("Initializing training algorithm")

    cost_model = Model(cost_network)
    value_model = Model(value_network)

    print "VALUE MODEL {}".format(
        value_model.get_parameter_values()['/policyandvaluea3c/mlp_value/linear_0.W'][0:10])


    print "COST MODEL {}".format(
        cost_model.get_parameter_values()['/policyandvaluea3c/mlp_value/linear_0.W'][0:10])

    
    if not async_update:
        algorithm = GradientDescent(
            cost=cost_network, parameters=cg.parameters,
            step_rule=Scale())
    else:
        print "CG.PARAMETERS", cg.parameters
        algorithm = AsyncUpdate(parameters=cg.parameters,
                                inputs=cost_model.get_parameter_dict().keys(),
                                step_rule=AsyncRMSProp(learning_rate=lr,
                                                       # FIXME: put as
                                                       # parameter
                                                       decay_rate=0.99,
                                                       max_scaling=10))

        # FIXME: delete not needed here
        # step_rule=CompositeRule([StepClipping(clip_c),
        #                         Adam(learning_rate=lr)]))

    algorithm.initialize()

    print "INPUTS ", cg.inputs
    f_cost = theano.function(inputs=cg.inputs, outputs=cg.outputs)
    f_policy = theano.function(inputs=cg_policy.inputs,
                               outputs=cg_policy.outputs)
    f_value = theano.function(inputs=cg_value.inputs, outputs=cg_value.outputs)

    f_extracost = theano.function(inputs=cg_extra.inputs,
                                  outputs=cg_extra.outputs)

    return cost_model, f_policy, f_value, algorithm, f_cost, f_extracost


if __name__ == "__main__":

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

    print "LEN ", len(policy_and_value_net.shared_a3c.layers)

    policy_and_value_net.shared_a3c.top_mlp.linear_transformations[
        0].weights_init = Uniform(width=.08)

    print "POLICY SPACE ", dir(policy_and_value_net.policy)
    print "VALUE SPACE ", dir(policy_and_value_net.value)

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

    #y = tensor.vector('targets')

    num_batches = 32
    zasca = np.array(np.random.randint(128, size=(num_batches, num_channels,
                                                  image_size[0],
                                                  image_size[1])),
                     dtype="uint8")

    pol_result = policy.eval({x: zasca})
    val_result = value.eval({x: zasca})

    # print "POLICY ", pol_result
    # print "VALUE ", val_result
    # print "ZASCA ",zasca

    # print "WEI ",dir(policy_and_value_net.shared_a3c.flattener.children)

    print "POLICY SHAPE ", np.shape(pol_result)
    print "VALUE SHAPE ", np.shape(val_result)
    print "ZASCA SHAPE ", np.shape(zasca)

    th_reward = T.matrix('ereward')
    th_actions = T.matrix('actions')

    reward = np.array(np.random.rand((num_batches)), dtype="float32")
    actions = np.arange(0, num_actions, dtype="int32")

    cost_network = policy_and_value_net.cost(x, th_actions, th_reward)
    cost_results = cost_network.eval(
        {x: zasca, th_actions: actions, th_reward: reward})

    print "COST ", cost_results
    print "COST SHAPE ", np.shape(cost_results)

    # Perform some optimization step
    cg = ComputationGraph(cost_network)

    if (dropout < 1.0):
        dropout_inputs = [x for x in cg.intermediary_variables]
        # print "DROPOUT INPUTS ",dropout_inputs
        #cg = apply_dropout(cg, dropout_inputs, dropout)

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
    print "COST MODEL ", cost_model.get_parameter_dict()

    print "DIR ", cost_model
    params = extract_params_from_model(cost_model)

    print "LEN PARAMS ", len(params)

    params_extra = OrderedDict()
    for kk in params:
        params_extra[kk] = params[kk] + 0.1

    update_cum_gradients(params, params_extra, cost_model)

    params_extra = OrderedDict()
    for kk in params:
        params_extra[kk] = params[kk] + 0.2

    update_cum_gradients(params, params_extra, cost_model)

    read_common_parameters(cost_model, cost_model)
