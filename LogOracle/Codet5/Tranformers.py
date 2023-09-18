import torch.nn as nn
import torch, math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class DenseLayer(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # nn.MultiHeadAttention will return a tuple includes output(idx=0) and attention weight(idx=1)
        return x + self.dropout(sublayer(self.norm(x))[0])
    
class EncodeTransformerBlock(nn.Module):
    '''
    Self Attention encoder for contextual inputs
    '''

    def __init__(self, hidden_dim, num_heads, feed_forward_hidden, dropout=0.1):
        super(EncodeTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(num_heads=num_heads, embed_dim=hidden_dim, batch_first=True,
                                               dropout=dropout)
        # self.attention = MultiHeadedAttention(h=num_heads, d_model=hidden_dim)
        self.feed_forward = DenseLayer(d_model=hidden_dim, d_ff=feed_forward_hidden, dropout=dropout)
        self.sublayer = SublayerConnection(size=hidden_dim, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden_dim, dropout=dropout)

    def forward(self, inputs, mask):
        # Self Attention + Add&Norm
        outputs = self.sublayer(inputs,
                                lambda _x: self.attention.forward(query=_x, key=_x, value=_x, key_padding_mask=mask,
                                                                  need_weights=False))
        # Feed Forward + Add&Norm
        outputs = self.sublayer2(outputs, self.feed_forward)
        return outputs