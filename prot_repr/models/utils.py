import six
from functools import partial
from ivbase.utils.commons import is_callable

import torch
import torch.nn as nn


SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh',
                            'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}

OPTIMIZERS = {
    'adadelta': torch.optim.Adadelta,
    'adagrad': torch.optim.Adagrad,
    'adam': torch.optim.Adam,
    'sparseadam': torch.optim.SparseAdam,
    'asgd': torch.optim.ASGD,
    'sgd': torch.optim.SGD,
    'rprop': torch.optim.Rprop,
    'rmsprop': torch.optim.RMSprop,
    'optimizer': torch.optim.Optimizer,
    'lbfgs': torch.optim.LBFGS
}


class GlobalMaxPool1d(nn.Module):
    r"""
    Global max pooling of a Tensor over one dimension
    See: https://stats.stackexchange.com/q/257321/

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.nn.commons.GlobalAvgPool1d`, :class:`ivbase.nn.commons.GlobalSumPool1d`

    """

    def __init__(self, dim=1):
        super(GlobalMaxPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        Arguments
        ----------
            x: torch.FloatTensor
                Input tensor

        Returns
        -------
            x: `torch.FloatTensor`
                Tensor resulting from the pooling operation.
        """
        return torch.max(x, dim=self.dim)[0]


class GlobalAvgPool1d(nn.Module):
    r"""
    Global Average pooling of a Tensor over one dimension

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.nn.commons.GlobalAvgPool1d`, :class:`ivbase.nn.commons.GlobalSumPool1d`
    """

    def __init__(self, dim=1):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            For more information, see :func:`ivbase.nn.commons.GlobalMaxPool1d.forward`

        """
        return torch.mean(x, dim=self.dim)


class GlobalSumPool1d(nn.Module):
    r"""
    Global Sum pooling of a Tensor over one dimension

    Arguments
    ----------
        dim: int, optional
            The dimension on which the pooling operation is applied.
            (Default value = 0)

    Attributes
    ----------
        dim: int
            The dimension on which the pooling operation is applied.

    See Also
    --------
        :class:`ivbase.nn.commons.GlobalMaxPool1d`, :class:`ivbase.nn.commons.GlobalAvgPool1d`

    """

    def __init__(self, dim=1):
        super(GlobalSumPool1d, self).__init__()
        self.dim = dim

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            For more information, see :func:`ivbase.nn.commons.GlobalMaxPool1d.forward`
        """
        return torch.sum(x, dim=self.dim)


class GlobalSoftAttention(nn.Module):
    r"""
    Global soft attention layer for computing the output vector of a graph convolution.
    It's akin to doing a weighted sum at the end

    Arguments
    ----------
        input_dim: int
            The input dimension of extracted features per nodes
        output_dim : int, optional
            The dimension on which all nodes will be projected
        use_sigmoid: bool, optional
            Use sigmoid instead of softmax. With sigmoid, the attention weights will not sum to 1.
            (Default value=False)

    See Also
    --------
        :class:`ivbase.nn.commons.GlobalMaxPool1d`, :class:`ivbase.nn.commons.GlobalAvgPool1d`

    """

    def __init__(self, input_dim, output_dim=None, use_sigmoid=False, **kwargs):
        super(GlobalSoftAttention, self).__init__()
        # Setting from the paper
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not self.output_dim:
            self.output_dim == self.input_dim

        # Embed graphs
        sig = nn.Sigmoid(-2) if use_sigmoid else nn.Softmax(-2)
        self.node_gating = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            sig
        )
        self.pooling = get_pooling("sum", dim=-2)
        self.graph_pooling = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            See :func:`ivbase.nn.commons.GlobalSumPool1d.forward`
        """
        if x.shape[0] == 0:
            return torch.zeros(1, self.output_dim)
        else:
            x =  torch.mul(self.node_gating(x), self.graph_pooling(x))
            return self.pooling(x)


class GlobalGatedPool(nn.Module):
    r"""
    Gated global pooling layer for computing the output vector of a graph convolution.

    Arguments
    ----------
        input_dim: int
            The input dimension of extracted features per nodes
        output_dim : int, optional
            The dimension on which all nodes will be projected
        dim: int, optional:
            The dimension on which feature will be aggregated. Default to second dimension for nodes.
            (Default value = -2)
        dropout: float, optional
            Whether dropout should be applied on the output
            (Default value=0)

    See Also
    --------
        :class:`ivbase.nn.commons.GlobalMaxPool1d`, :class:`ivbase.nn.commons.GlobalAvgPool1d`, :class:`ivbase.nn.commons.GlobalSoftAttention`

    """

    def __init__(self, input_dim, output_dim, dim=-2, dropout=0., **kwargs):
        super(GlobalGatedPool, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.dim = dim
        self.sigmoid_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                            nn.Sigmoid())
        self.tanh_linear = nn.Sequential(nn.Linear(self.input_dim, self.output_dim),
                                         nn.Tanh())
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        r"""
        Applies the pooling on input tensor x

        See Also
        --------
            See :func:`ivbase.nn.commons.GlobalSumPool1d.forward`
        """
        i = self.sigmoid_linear(input)
        j = self.tanh_linear(input)
        output = torch.sum(torch.mul(i, j), self.dim)  # on the node dimension
        output = self.dropout(output)
        return output


class Transpose(nn.Module):
    r"""
    Transpose two dimensions of some tensor.

    Arguments
    ----------
        dim1: int
            First dimension concerned by the transposition
        dim2: int
            Second dimension concerned by the transposition

    Attributes
    ----------
        dim1: int
            First dimension concerned by the transposition
        dim2: int
            Second dimension concerned by the transposition
    """

    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        r"""
        Transposes dimension of input x

        Arguments
        ----------
            x: torch.Tensor
                Tensor to be transpose. x should support dim1 and dim2.

        Returns
        -------
            x: torch.Tensor
                transposed version of input

        """
        return x.transpose(self.dim1, self.dim2)


class Chomp1d(nn.Module):
    r"""
    Chomp or trim a batch of sequences represented as 3D tensors

    Arguments
    ----------
        chomp_size: int
            the length of the sequences after the trimming operation

    Attributes
    ----------
        chomp_size: int
            sequence length after trimming
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        r"""
        Trim input x

        Arguments
        ----------
            x: 3D torch.Tensor
                batch of sequences represented as 3D tensors, the first dim is the batch size,
                the second is the length of the sequences, and the last their embedding size

        Returns
        -------
            x: 3D torch.Tensor
                New tensor which second dimension
                has been modified according to the chomp_size specified
        """
        return x[:, :-self.chomp_size, :].contiguous()


class ResidualBlock(nn.Module):
    r"""Residual Block maker
    Let :math:`f` be a module, the residual block acts as a module g such as :math:`g(x) = \text{ReLU}(x + f(x))`

    Arguments
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.nn.Module, optional
            A down/up sampling module, which is needed
            when the output of the base_module doesn't lie in the same space as its input.
            (Default value = None)
        auto_sample: bool, optional
            Whether to force resampling when the input and output
            dimension of the base_module do not match, and no resampling module was provided.
            By default, the `torch.nn.functional.interpolate` function will be used.
            (Default value = False)
        activation: str or callable
            activation function to use for the residual block
            (Default value = 'relu')
        kwargs: named parameters for the `torch.nn.functional.interpolate` function

    Attributes
    ----------
        base_module: torch.nn.Module
            The module that will be made residual
        resample: torch.nn.Module
            the resampling module
        interpolate: bool
            specifies if resampling should be enforced.

    """

    def __init__(self, base_module, resample=None, auto_sample=False, activation='relu', **kwargs):
        super(ResidualBlock, self).__init__()
        self.base_module = base_module
        self.resample = resample
        self.interpolate = False
        self.activation = get_activation(activation)
        if resample is None and auto_sample:
            self.resample = partial(nn.functional.interpolate, **kwargs)
            self.interpolate = True

    def forward(self, x):
        r"""
        Applies residual block on input tensor.
        The output of the base_module will be automatically resampled
        to match the input

        Arguments
        ----------
            x: torch.Tensor
                The input of the residual net

        Returns
        -------
            out: torch.Tensor
                The output of the network
        """
        residual = x
        indim = residual.shape[-1]
        out = self.base_module(x)
        outdim = out.shape[-1]
        if self.resample is not None and not self.interpolate:
            residual = self.resample(x)
        elif self.interpolate and outdim != indim:
            residual = self.resample(x, size=outdim)

        out += residual
        if self.activation:
            out = self.activation(out)
        return out


def get_activation(activation):
    r"""
    Get a pytorch activation layer based on its name. This function acts as a shortcut
    and is case insensitive. Implemented layers should use this function as an activation provider.
    When the input value is callable and not a string, it is assumed that it corresponds to the
    activation function, and thus returned as is.

    Arguments
    ---------
        activation: str or callable
            a python callable or the name of the activation function

    Returns
    -------
        act_cls: torch.nn.Module instance
            module that represents the activation function.
    """
    if is_callable(activation):
        return activation
    activation = [
        x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) > 0 and isinstance(activation[0], six.string_types), \
        'Unhandled activation function'
    if activation[0].lower() == 'none':
        return None

    return vars(torch.nn.modules.activation)[activation[0]]()


def get_pooling(pooling, **kwargs):
    r"""
    Get a pooling layer by name. The recommended way to use a pooling layer is through
    this function. When the input is a callable and not a string, it is assumed that it
    corresponds to the pooling layer, and thus returned as is.

    Arguments
    ----------
        pooling: str or callable
            a python callable or the name of the activation function
            Supported pooling functions are 'avg', 'sum' and 'max'
        kwargs:
            Named parameters values that will be passed to the pooling function

    Returns
    -------
        pool_cls: torch.nn.Module instance
            module that represents the activation function.
    """
    if is_callable(pooling):
        return pooling
    # there is a reason for this to not be outside
    POOLING_MAP = {"max": GlobalMaxPool1d, "avg": GlobalAvgPool1d,
                   "sum": GlobalSumPool1d, "mean": GlobalAvgPool1d, 'attn': GlobalSoftAttention, 'attention': GlobalSoftAttention, 'gated': GlobalGatedPool}
    return POOLING_MAP[pooling.lower()](**kwargs)


def get_optimizer(optimizer):
    r"""
    Get an optimizer by name. cUstom optimizer, need to be subclasses of :class:`torch.optim.Optimizer`.

    Arguments
    ----------
        optimizer: :class:`torch.optim.Optimizer` or str
            A class (not an object) or a valid pytorch Optimizer name

    Returns
    -------
        optm `torch.optim.Optimizer`
            Class that should be initialized to get an optimizer.s
    """
    if not isinstance(optimizer, six.string_types) and issubclass(optimizer, torch.optim.Optimizer):
        return optimizer
    return OPTIMIZERS[optimizer.lower()]


