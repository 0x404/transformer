import math
import torch
import torch.nn as nn


class DotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        """Perform sacaled dot product attention.

        Args:
            query (Tensor): with shape [*, len, n_model]
            key (Tensor): with shape [*, len, n_model]
            value (Tensor): with shape [*, len, n_model]
            mask (Tensor, optional): mask tensor. Defaults to None.
            dropout (callable, optional): dropout funciton. Defaults to None.
        """

        # consider three matrices of the same size [len, n_model] as Q, K and V
        # we calculate their similarity by dot product
        # i.e. Q[len, n_model] * K_T[n_model, len] -> R[len, len]
        # where R[i, j] means similarity between Q[i] and K[j]
        result = torch.matmul(query, key.transpose(-2, -1))
        result = result / math.sqrt(query.size(-1))

        if mask is not None:
            result = result.masked_fill(mask == 0, 1e-9)

        # we do softmax for each row in R, then R[i] represents the distance
        # between the i-th qurey and all keys
        result = torch.nn.functional.softmax(result, dim=-1)

        if dropout is not None:
            result = dropout(result)

        # finnaly we do matmul of R and V
        return torch.matmul(result, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head: int, d_model: int, dropout: float = 0.1):
        """Multi head attention module.

        Args:
            num_head (int): the number of head, must be divisible by d_model.
            d_model (int): the dimension of word vector.
            dropout (float): dropout probility.
        """
        super().__init__()
        assert d_model % num_head == 0, "num_head must be divisible by d_model"

        self.num_head = num_head
        self.d_head = d_model // num_head

        # the reason we only create 3 linears for q, k, v is that
        # we can reduce the number of matmul and speed up the operation

        # intutive implementation of multi head attention should be:
        # we create 3 * h linear(d_model, d_model / num_head)
        # to map `d_model` dimension to `d_model / num_head` dimension
        # and we shuld do 3 * h matmuls

        # howerver, the number of learned parameters for theses two impls
        # is completely equivalent
        # as for why we don't reduce the number of matmul to 1
        # it's just because we store `query`, `key`, `value` separately
        # we can just create a linear(3 * d_model, 3 * d_model)
        # if they are stored in one tensor
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.attention = DotProductAttention()
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        Perform multi head attention.

        Args:
            query (Tensor): with shape [batch, len, d_model].
            key (Tensor): with shape [batch, len, d_model].
            value (Tensor): with shape [batch, len, d_model].
            mask (Tensor, optional): mask Tensor. Defaults to None.

        Returns:
            Tersor: with shape [batch, len, d_model].
        """
        batch_size = query.size(0)

        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # extract num_head dimension
        query = query.view(batch_size, -1, self.num_head, self.d_head)
        key = key.view(batch_size, -1, self.num_head, self.d_head)
        value = value.view(batch_size, -1, self.num_head, self.d_head)

        # make shape to [*, len, d_head]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # attn's shape is [batch_size, num_head, len, d_head]
        attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # reshape attn's shape to [batch_size, len, d_model]
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_head * self.d_head)
        )

        # remap d_head dimension to d_model dimension
        attn = self.out_linear(attn)
        return attn


if __name__ == "__main__":
    multi_attn = MultiHeadedAttention(num_head=2, d_model=10)
    batch_data = torch.rand(3, 5, 10)
    result = multi_attn(batch_data, batch_data, batch_data)
    assert batch_data.shape == result.shape
