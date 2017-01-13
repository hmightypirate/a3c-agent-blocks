"""Training algorithms."""
import logging
import itertools
from collections import OrderedDict

from picklable_itertools.extras import equizip
import numpy as np
import theano
from theano import tensor
from blocks.roles import add_role, ALGORITHM_HYPERPARAMETER
from blocks.theano_expressions import l2_norm
from blocks.algorithms import (StepRule, Scale,
                               _create_algorithm_buffer_for,
                               CompositeRule, TrainingAlgorithm)
from blocks.utils import (shared_floatx)

logger = logging.getLogger(__name__)


class AsyncBasicRMSProp(StepRule):
    """Scales the step size by a running average of the recent step norms.

    Parameters
    ----------
    decay_rate : float, optional
        How fast the running average decays, value in [0, 1]
        (lower is faster).  Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Needs to be greater than 0. Defaults to 1e5.

    Notes
    -----
    This step rule is intended to be used in conjunction with another
    step rule, _e.g._ :class:`Scale`. For an all-batteries-included
    experience, look at :class:`RMSProp`.

    In general, this step rule should be used _before_ other step rules,
    because it has normalization properties that may undo their work.
    For instance, it should be applied first when used in conjunction
    with :class:`Scale`.

    For more information, see [Hint2014]_.

    """

    def __init__(self, decay_rate=0.9, max_scaling=1e5):
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError("decay rate needs to be in [0, 1]")
        if max_scaling <= 0:
            raise ValueError("max. scaling needs to be greater than 0")
        self.decay_rate = shared_floatx(decay_rate, "decay_rate")
        add_role(self.decay_rate, ALGORITHM_HYPERPARAMETER)
        self.epsilon = 1. / max_scaling

    def compute_step(self, parameter, previous_step):
        mean_square_step_tm1 = _create_algorithm_buffer_for(
            parameter, "mean_square_step_tm1")
        mean_square_step_t = (
            self.decay_rate * mean_square_step_tm1 +
            (1 - self.decay_rate) * tensor.sqr(previous_step))
        rms_step_t = tensor.sqrt(mean_square_step_t + self.epsilon)
        step = previous_step / rms_step_t
        updates = [(mean_square_step_tm1, mean_square_step_t)]
        return step, updates


class AsyncRMSProp(CompositeRule):
    """Scales the step size by a running average of the recent step norms.

    Combines :class:`BasicRMSProp` and :class:`Scale` to form the step rule
    described in [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    Parameters
    ----------
    learning_rate : float, optional
        The learning rate by which the previous step scaled. Defaults to 1.
    decay_rate : float, optional
        How fast the running average decays (lower is faster).
        Defaults to 0.9.
    max_scaling : float, optional
        Maximum scaling of the step size, in case the running average is
        really small. Defaults to 1e5.

    Attributes
    ----------
    learning_rate : :class:`~tensor.SharedVariable`
        A variable for learning rate.
    decay_rate : :class:`~tensor.SharedVariable`
        A variable for decay rate.

    See Also
    --------
    :class:`SharedVariableModifier`

    """

    def __init__(self, learning_rate=1.0, decay_rate=0.9, max_scaling=1e5):
        basic_rms_prop = AsyncBasicRMSProp(decay_rate=decay_rate,
                                           max_scaling=max_scaling)
        scale = Scale(learning_rate=learning_rate)
        self.learning_rate = scale.learning_rate
        self.decay_rate = basic_rms_prop.decay_rate
        self.components = [basic_rms_prop, scale]


class AsyncUpdatesAlgorithm(TrainingAlgorithm):
    """Base class for async algorithms that use Theano functions with updates.

    Parameters
    ----------
    inputs : TODO
    updates : list of tuples or :class:`~collections.OrderedDict`
        The updates that should be performed.
    theano_func_kwargs : dict, optional
        A passthrough to `theano.function` for additional arguments.
        Useful for passing `profile` or `mode` arguments to the theano
        function that will be compiled for the algorithm.
    on_unused_sources : str, one of 'raise' (default), 'ignore', 'warn'
        Controls behavior when not all sources in a batch are used
        (i.e. there is no variable with a matching name in the inputs
        of the computational graph of the updates).

    Attributes
    ----------
    updates : list of :class:`~tensor.TensorSharedVariable` updates
        Updates to be done for every batch. It is required that the
        updates are done using the old values of optimized parameters.

    Notes
    -----
    Changing `updates` attribute or calling `add_updates` after
    the `initialize` method is called will have no effect.

    """

    def __init__(self, inputs=None,  updates=None, theano_func_kwargs=None,
                 on_unused_sources='raise', **kwargs):
        self.updates = [] if updates is None else updates
        self.theano_func_kwargs = (theano_func_kwargs if theano_func_kwargs
                                   is not None else dict())
        self.inputs = inputs

        print "SELF INPUTS ", inputs
        self.on_unused_sources = on_unused_sources
        super(AsyncUpdatesAlgorithm, self).__init__(**kwargs)

    def initialize(self):
        logger.info("Initializing the training algorithm")
        # update_values = [new_value for _, new_value in self.updates]
        logger.debug("Inferring graph inputs...")
        logger.debug("Compiling training function...")
        self._function = theano.function(
            list(self.inputs), [], updates=self.updates,
            **self.theano_func_kwargs)
        logger.info("The training algorithm is initialized")

    @property
    def updates(self):
        return self._updates

    @updates.setter
    def updates(self, value):
        self._updates = value

    def add_updates(self, updates):
        """Add updates to the training process.

        The updates will be done _before_ the parameters are changed.

        Parameters
        ----------
        updates : list of tuples or :class:`~collections.OrderedDict`
            The updates to add.

        """
        if isinstance(updates, OrderedDict):
            updates = list(updates.items())
        if not isinstance(updates, list):
            raise ValueError
        self.updates.extend(updates)

    def _validate_source_names(self, batch):
        in_names = [v.name for v in self.inputs]

        if not set(in_names).issubset(set(batch.keys())):
            raise ValueError("Didn't find all sources: " +
                             source_missing_error.format(
                                 sources=batch.keys(),
                                 variables=in_names))
        if not set(batch.keys()).issubset(set(in_names)):
            if self.on_unused_sources == 'ignore':
                pass
            elif self.on_unused_sources == 'warn':
                if not hasattr(self, '_unused_source_warned'):
                    logger.warn(variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=in_names))
                self._unused_source_warned = True
            elif self.on_unused_sources == 'raise':
                raise ValueError(
                    "mismatch of variable names and data sources" +
                    variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=in_names))
            else:
                raise ValueError("Wrong value of on_unused_sources: {}."
                                 .format(self.on_unused_sources))

    def process_batch(self, batch):
        self._validate_source_names(batch)
        ordered_batch = [batch[v.name] for v in self.inputs]
        self._function(*ordered_batch)


class AsyncUpdate(AsyncUpdatesAlgorithm):
    """ A base class for async updates

    TODO
    """

    def __init__(self, parameters=None, inputs=None, step_rule=None, **kwargs):
        # set initial parameters
        self.parameters = parameters
        self.inputs = inputs
        # This is the gradient of the parameter which
        # has the same size as parameters
        print "parameters ", parameters

        previous_steps = [
            tensor.TensorType(dtype="float32",
                              broadcastable=np.zeros_like(
                                  p.get_value().shape))(v)
            for p, v in equizip(self.parameters, self.inputs)]

        inputs = previous_steps

        previous_steps = OrderedDict(equizip(self.parameters, previous_steps))

        self.step_rule = step_rule if step_rule else Scale()
        logger.debug("Computing parameter steps...")
        self.steps, self.step_rule_updates = (self.step_rule.compute_steps(
            previous_steps))

        # Same as gradient_values above: the order may influence a
        # bunch of things, so enforce a consistent one (don't use
        # .values()).
        step_values = [self.steps[p] for p in self.parameters]
        self.total_step_norm = (l2_norm(step_values))

        # Once again, iterating on gradients may not be deterministically
        # ordered if it is not an OrderedDict. We add the updates here in
        # the order specified in self.parameters. Keep it this way to
        # maintain reproducibility.
        kwargs.setdefault('updates', []).extend(
            itertools.chain(((parameter, parameter - self.steps[parameter])
                             for parameter in self.parameters),
                            self.step_rule_updates))

        super(AsyncUpdate, self).__init__(inputs=inputs, **kwargs)
