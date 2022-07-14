import torch.nn as nn
from .layers import PositionwiseFeedForward, AddNorm
from .attention import MultiHeadedAttention
from .embeding import PositionalEmbedding


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_head, d_ff=512, dropout=0.1):
        """Transformer encoder block.

        Args:
            d_model (int): dimension of word vector.
            num_head (int): number of attention heads.
            d_ff (int, optional): size of hidden layer in ff layer. Defaults to 512.
            dropout (float, optional): dropout probility. Defaults to 0.1.
        """
        super().__init__()
        self.attn = MultiHeadedAttention(d_model, num_head, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)

    def forward(self, x, mask=None):
        y = self.attn(x, x, x, mask=mask)
        y = self.addnorm1(x, y)
        z = self.ffn(y)
        z = self.addnorm2(y, z)
        return z


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_enc, vocab_size, d_model, num_head, d_ff=512, dropout=0.1, max_len=512
    ):
        """Transformer encoder.

        Args:
            num_enc (int): number of transformer encoder block.
            vocab_size (int): vocabulary size.
            d_model (int): dimension of word vector.
            num_head (int): number of attention heads.
            d_ff (int, optional): size of hidden layer in ff layer. Defaults to 512.
            dropout (float, optional): dropout probility. Defaults to 0.1.
            max_len (int, optional): max length of a sentence. Defaults to 512.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEmbedding(d_model, max_len=max_len)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_head, d_ff=d_ff, dropout=dropout)
                for _ in range(num_enc)
            ]
        )

    def forward(self, x, mask):
        """Transformer encoder forward.

        Args:
            x (Tensor): origin input, with shape [batch, len, 1].
            mask (Tensor): mask tensor.

        Returns:
            Tensor: with shape [batch, len, d_model]
        """
        x = self.embed(x)
        x = self.pos_enc(x)
        for enc in self.encoders:
            x = enc(x, mask=mask)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_head, d_ff=512, dropout=0.1):
        """Transformer decoder block.

        Args:
            d_model (int): dimension of word vector.
            num_head (int): number of attention heads.
            d_ff (int, optional): size of hidden layer in ff layer. Defaults to 512.
            dropout (float, optional): dropout probility. Defaults to 0.1.
        """
        super().__init__()
        self.attn1 = MultiHeadedAttention(d_model, num_head, dropout=dropout)
        self.addnorm1 = AddNorm(d_model, dropout=dropout)
        self.attn2 = MultiHeadedAttention(d_model, num_head, dropout=dropout)
        self.addnorm2 = AddNorm(d_model, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.addnorm3 = AddNorm(d_model, dropout=dropout)

    def forward(self, x, y, x_mask, y_mask):
        """Transformer decoder block forward.

        Args:
            x (Tensor): output embed, with shape [batch, len, d_model].
            y (Tensor): encoder output, with shape [batch, len, d_model].
            x_mask (Tensor): output embeding mask.
            y_mask (Tensor): encoder output mask.

        Returns:
            Tensor: with shape [batchm len, d_model].
        """
        x1 = self.attn1(x, x, x, mask=x_mask)
        x2 = self.addnorm1(x, x1)
        x3 = self.attn2(x2, y, y, mask=y_mask)
        x4 = self.addnorm2(x2, x3)
        x5 = self.ffn(x4)
        x6 = self.addnorm2(x4, x5)
        return x6


class TransformerDecoder(nn.Module):
    def __init__(
        self, num_dec, vocab_size, d_model, num_head, d_ff=512, dropout=0.1, max_len=512
    ):
        """Transformer decoder.

        Args:
            num_dec (int): number of decoders.
            vocab_size (int): vocabulary size.
            d_model (int): dimension of word vector.
            num_head (int): number of attention heads.
            d_ff (int, optional): size of hidden layer of ff lyaer. Defaults to 512.
            dropout (float, optional): dropout probility. Defaults to 0.1.
            max_len (int, optional): max length of a sentence. Defaults to 512.
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEmbedding(d_model, max_len=max_len)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_head, d_ff=d_ff, dropout=dropout)
                for _ in range(num_dec)
            ]
        )

    def forward(self, x, y, x_mask, y_mask):
        """Transformer decoder forward.

        Args:
            x (Tensor): outputs.
            y (Tensor): decoder outputs.
            x_mask (Tensor): outputs mask.
            y_mask (Tensor): decoder outputs mask.
        """
        x = self.embed(x)
        x = self.pos_enc(x)
        for enc in self.decoders:
            x = enc(x, y, x_mask=x_mask, y_mask=y_mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab,
        trg_vocab,
        d_model,
        num_head,
        num_enc,
        num_dec,
        d_ff=512,
        dropout=0.1,
        max_len=512,
    ):
        """Transformer module.

        Args:
            src_vocab (int): source vocabulary size.
            trg_vocab (int): traget vocabulary size.
            d_model (int): dimension of word vector.
            num_head (int): number of attension heads.
            num_enc (int): number of transformer encoder.
            num_dec (int): number of transformer decoder.
            d_ff (int, optional): size of hidden layer in ff layer. Defaults to 512.
            dropout (float, optional): dropout probility. Defaults to 0.1.
            max_len (int, optional): max length of a sentence. Defaults to 512.
        """
        super().__init__()
        self.encoder = TransformerEncoder(
            num_enc,
            src_vocab,
            d_model,
            num_head,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = TransformerDecoder(
            num_dec,
            trg_vocab,
            d_model,
            num_head,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.out_linear = nn.Linear(d_model, trg_vocab)

    def forward(self, inputs, outputs, inputs_mask, outputs_mask):
        """Transformer forward.

        Args:
            inputs (Tensor): inputs.
            outputs (Tensor): outputs.
            inputs_mask (Tensor): intputs mask.
            outputs_mask (Tensor): outputs mask.
        """
        encoder_out = self.encoder(inputs, mask=inputs_mask)
        decoder_out = self.decoder(outputs, encoder_out, outputs_mask, inputs_mask)
        out = self.out_linear(decoder_out)
        return out
