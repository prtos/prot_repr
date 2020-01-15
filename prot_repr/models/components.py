from torch.nn import AvgPool1d, BatchNorm1d, Conv1d, Dropout, Embedding, MaxPool1d, Sequential, Module, Linear
from prot_repr.models.attention import SelfAttention
from prot_repr.models.utils import Transpose,  get_activation


class Cnn1dEncoder(Module):
    r"""
    Extract features from a sequence-like data using Convolutional Neural Network.
    Each time step or position of the sequence must be a discrete value.

    Arguments
    ----------
        vocab_size: int
            Size of the vocabulary, i.e the maximum number of discrete elements possible at each time step.
            Since padding will be used for small sequences, we expect the vocab size to be 1 + size of the alphabet.
            We also expect that 0 won't be use to represent any element of the vocabulary expect the padding.
        embedding_size: int
            The size of each embedding vector
        cnn_sizes: int list
            A list that specifies the size of each convolution layer.
            The size of the list implicitly defines the number of layers of the network
        kernel_size: int or list(int)
            he size of the kernel, i.e the number of time steps include in one convolution operation.
            An integer means the same value will be used for each conv layer. A list allows to specify different sizes for different layers.
            The length of the list should match the length of cnn_sizes.
        pooling_len: int or int list, optional
            The number of time steps aggregated together by the pooling operation.
            An integer means the same pooling length is used for all layers.
            A list allows to specify different length for different layers. The length of the list should match the length of cnn_sizes
            (Default value = 1)
        pooling: str, optional
            One of {'avg', 'max'} (for AveragePooling and MaxPooling).
            It indicates the type of pooling operator to use after convolution.
            (Default value = 'avg')
        dilatation_rate: int or int list, optional
            The dilation factor tells how large are the gaps between elements in
            a feature map on which we apply a convolution filter.  If a integer is provided, the same value is used for all
            convolution layer. If dilation = 1 (no gaps),  every 1st element next to one position is included in the conv op.
            If dilation = 2, we take every 2nd (gaps of size 1), and so on. See https://arxiv.org/pdf/1511.07122.pdf for more info.
            (Default value = 1)
        activation: str or callable, optional
            The activation function. activation layer {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus'}
            The name of the activation function
        b_norm: bool, optional
            Whether to use Batch Normalization after each convolution layer.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, vocab_size, embedding_size, cnn_sizes, output_size,
                 kernel_size, pooling_len=1, pooling='avg',
                 dilatation_rate=1, activation='ReLU', b_norm=False,
                 dropout=0.0, padding_idx=None):
        super(Cnn1dEncoder, self).__init__()
        self.__params = locals()
        activation_cls = get_activation(activation)
        if not isinstance(pooling_len, (list, int)):
            raise TypeError("pooling_len should be of type int or int list")
        if pooling not in ['avg', 'max']:
            raise ValueError("the pooling type must be either 'max' or 'avg'")
        if len(cnn_sizes) <= 0:
            raise ValueError(
                "There should be at least on convolution layer (cnn_size should be positive.)")

        if isinstance(pooling_len, int):
            pooling_len = [pooling_len] * len(cnn_sizes)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(cnn_sizes)
        if pooling == 'avg':
            pool1d = AvgPool1d
        else:
            pool1d = MaxPool1d

        # network construction
        embedding = Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        layers = [Transpose(1, 2)]
        in_channels = [embedding_size] + cnn_sizes[:-1]
        for i, (in_channel, out_channel, ksize, l_pool) in \
                enumerate(zip(in_channels, cnn_sizes, kernel_size, pooling_len)):
            pad = ((dilatation_rate**i) * (ksize - 1) + 1) // 2
            layers.append(Conv1d(in_channel, out_channel, padding=pad,
                                 kernel_size=ksize, dilation=dilatation_rate**i))
            if b_norm:
                layers.append(BatchNorm1d(out_channel))
            layers.append(activation_cls)
            layers.append(Dropout(dropout))
            if l_pool > 1:
                layers.append(pool1d(l_pool))

        layers.append(Transpose(1, 2))

        self.locals_extractor = Sequential(embedding, *layers)
        self.global_pooler = SelfAttention(cnn_sizes[-1], output_size, pooling=pooling)
        self.__g_output_dim =output_size
        self.__l_output_dim = cnn_sizes[-1]


    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the sequences are projected

        Returns
        -------
        output_dim (int): Dimension of the output feature space

        """
        return self.__g_output_dim

    @property
    def local_output_dim(self):
        r"""
        Get the dimension of the feature space in which the sequences are projected

        Returns
        -------
        output_dim (int): Dimension of the output feature space

        """
        return self.__l_output_dim

    def forward(self, x, return_locals=False):
        r"""
        Forward-pass method

        Arguments
        ----------
            x (torch.LongTensor of size N*L): Batch of N sequences of size L each.
                L is actually the length of the longest of the sequence in the bacth and we expected the
                rest of the sequences to be padded with zeros up to that length.
                Each entry of the tensor is supposed to be an integer representing an element of the vocabulary.
                0 is reserved as the padding marker.

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors. D is the dimension of the feature space.
                D is given by output_dim.
        """
        locals = self.locals_extractor(x)
        globals_ = self.global_pooler(locals)
        if return_locals:
            return globals_, locals
        return globals_


class FCLayer(Module):
    r"""
    A simple fully connected and customizable layer. This layer is centered around a torch.nn.Linear module.
    The order in which transformations are applied is:

    #. Dense Layer
    #. Activation
    #. Dropout (if applicable)
    #. Batch Normalization (if applicable)

    Arguments
    ----------
        in_size: int
            Input dimension of the layer (the torch.nn.Linear)
        out_size: int
            Output dimension of the layer. Should be one supported by :func:`ivbase.nn.commons.get_activation`.
        dropout: float, optional
            The ratio of units to dropout. No dropout by default.
            (Default value = 0.)
        activation: str or callable, optional
            Activation function to use.
            (Default value = relu)
        b_norm: bool, optional
            Whether to use batch normalization
            (Default value = False)
        bias: bool, optional
            Whether to enable bias in for the linear layer.
            (Default value = True)
        init_fn: callable, optional
            Initialization function to use for the weight of the layer. Default is
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` with :math:`k=\frac{1}{ \text{in_size}}`
            (Default value = None)

    Attributes
    ----------
        dropout: int
            The ratio of units to dropout.
        b_norm: int
            Whether to use batch normalization
        linear: torch.nn.Linear
            The linear layer
        activation: the torch.nn.Module
            The activation layer
        init_fn: function
            Initialization function used for the weight of the layer
        in_size: int
            Input dimension of the linear layer
        out_size: int
            Output dimension of the linear layer

    """

    def __init__(self, in_size, out_size, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None):
        super(FCLayer, self).__init__()
        # Although I disagree with this it is simple enough and robust
        # if we trust the user base
        self._params = locals()
        activation = get_activation(activation)
        linear = Linear(in_size, out_size, bias=bias)
        if init_fn:
            init_fn(linear)
        layers = [linear, activation]
        if dropout:
            layers.append(Dropout(p=dropout))
        if b_norm:
            layers.append(BatchNorm1d(out_size))
        self.net = Sequential(*layers)


    @property
    def output_dim(self):
        r"""
        Dimension of the output feature space in which the input are projected

        Returns
        -------
            output_dim (int): Output dimension of this layer

        """
        return self.out_size

    def forward(self, x):
        r"""
        Compute the layer transformation on the input.

        Arguments
        ----------
            x: torch.Tensor
                input variable

        Returns
        -------
            out: torch.Tensor
                output of the layer
        """
        return self.net(x)


class FcNet(Module):
    r"""
    Fully Connected Neural Network

    Arguments
    ----------
        input_size: int
            size of the input
        hidden_sizes: int list or int
            size of the hidden layers
        activation: str or callable
            activation function. Should be supported by :func:`ivbase.nn.commons.get_activation`
            (Default value = 'relu')
        b_norm: bool, optional):
            Whether batch norm is used or not.
            (Default value = False)
        dropout: float, optional
            Dropout probability to regularize the network. No dropout by default.
            (Default value = .0)

    Attributes
    ----------
        extractor: torch.nn.Module
            The underlying feature extractor of the model.
    """

    def __init__(self, input_size, hidden_sizes, activation='ReLU',
                 b_norm=False, dropout=0.0, ):
        super(FcNet, self).__init__()
        self._params = locals()
        layers = []
        in_ = input_size
        for i, out_ in enumerate(hidden_sizes):
            layer = FCLayer(in_, out_, activation=activation,
                            b_norm=b_norm and (i == (len(hidden_sizes) - 1)),
                            dropout=dropout)
            layers.append(layer)
            in_ = out_

        self.__output_dim = in_
        self.extractor = Sequential(*layers)

    @property
    def output_dim(self):
        r"""
        Get the dimension of the feature space in which the elements are projected

        Returns
        -------
        output_dim: int
            Dimension of the output feature space
        """
        return self.__output_dim

    def forward(self, x):
        r"""
        Forward-pass method

        Arguments
        ----------
            x: torch.FloatTensor of size N*L
                Batch of N input vectors of size L (input_size).

        Returns
        -------
            phi_x: torch.FloatTensor of size N*D
                Batch of feature vectors.
                D is the dimension of the feature space (the model's output_dim).
        """
        return self.extractor(x)


def get_encoder(arch, *args, **kwargs):
    if arch.lower() == 'cnn':
        return Cnn1dEncoder(*args, **kwargs)
    elif arch.lower() == 'lstm':
        raise Exception('Not implemented yet')
    else:
        raise Exception("")