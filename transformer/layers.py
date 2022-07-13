import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Position wise feed forward.

        Args:
            x (Tensor): with shape [batch, len, d_model].

        Returns:
            Tensor: with shape [batch, len, d_model].
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.larer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """Add and layer normalization.

        Args:
            x (Tensor): original input, with shape [batch, len, d_model].
            y (Tensor): modle output, with shape [batch, len, d_model].

        Returns:
            Tensor: layernorm(x + dropout(y)), with shape [batch, len, d_model]
        """
        y = self.dropout(y)
        x = x + y
        x = self.larer_norm(x)
        return x
