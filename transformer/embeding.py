import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Positional embedding module.
    Add timing information to a sequence.

    Args:
        d_model (int): dimension of word vector.
        max_len (int): max length of a sentence.
    """

    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float, requires_grad=False)
        div_term = [10000 ** ((2 * i) / d_model) for i in range(0, d_model, 2)]

        # pe(pos, 2i) = sin(pos / (10000 ** (2i / d_model)))
        # pe(pos, 2i + 1) = cos(pos / (10000 ** (2i / d_model)))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / div_term[i // 2])
            for i in range(1, d_model, 2):
                pe[pos, i] = math.cos(pos / div_term[i // 2])

        # make shape of pe to [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, scaled: bool = True):
        """Add positional embedding to input x.

        Args:
            x (torch.Tensor): with shape [batch, len, d_model]
            scaled (bool): whther scale x by sqrt(d_model), default True.
        """
        if scaled:
            x = x * math.sqrt(x.size(-1))
        x = x + self.pe[:, : x.size(1)]
        return x


if __name__ == "__main__":
    pos_embed = PositionalEmbedding(4, 5)
    print(pos_embed.pe)
    x = torch.zeros(2, 3, 4)
    x = pos_embed(x, False)
    print(x)
