import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ivbase.nn.commons import get_pooling


class SelfAttention(nn.Module):
    r"""
    Standard Self Attention module to emulate interactions between elements of a sequence

    Arguments
    ----------
        input_size: int
            Size of the input vector at each time step
        output_size: int
            Size of the output at each time step
        outnet: Union[`torch.nn.module`, callable], optional:
            Neural network that will predict the output. If not provided,
            A MLP without activation will be used.
            (Default value = None)
        pooling: str, optional
            Pooling operation to perform. It can be either
            None, meaning no pooling is performed, or one of the supported pooling
            function name (see :func:`ivbase.nn.commons.get_pooling`)
            (Default value = None)

    Attributes
    ----------
        attention_net:
            linear function to use for computing the attention on input
        output_net:
            linear function for computing the output values on
            which attention should be applied
    """

    def __init__(self, input_size, output_size, outnet=None, pooling=None):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.attention_net = nn.Linear(self.input_size, self.input_size, bias=False)
        self.output_net = outnet or nn.Linear
        self.output_net = self.output_net(self.input_size, self.output_size)
        self.pooling = None
        if pooling:
            self.pooling = get_pooling(pooling)
            # any error here should be propagated immediately

    def forward(self, x, value=None, return_attention=False):
        r"""
        Applies attention on input

        Arguments
        ----------
            x: torch.FLoatTensor of size B*N*M
                Batch of B sequences of size N each.with M features.Note that M must match the input size vector
            value: torch.FLoatTensor of size B*N*D, optional
                Use provided values, instead of computing them again. This is to address case where the output_net has complex input.
                (Default value = None)
            return_attention: bool, optional
                Whether to return the attention matrix.

        Returns
        -------
            res: torch.FLoatTensor of size B*M' or B*N*M'
                The shape of the resulting output, will depends on the presence of a pooling operator for this layer
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        assert x.size(-1) == self.input_size
        query = x
        key = self.attention_net(x)
        if value is None:
            value = self.output_net(x)
        key = key.transpose(1, 2)
        attention_matrix = torch.bmm(query, key)
        attention_matrix = attention_matrix / math.sqrt(self.input_size)
        attention_matrix = F.softmax(attention_matrix, dim=2)
        applied_attention = torch.bmm(attention_matrix, value)
        if self.pooling is None:
            res = applied_attention
        else:
            res = self.pooling(applied_attention)
        if return_attention:
            return res, attention_matrix
        return res

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'insize=' + str(self.input_size) \
            + ', outsize=' + str(self.output_size) + ')'


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.

    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}

    Args:
        dim(int): The number of expected features in the output

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    Examples::

         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)

    """
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
