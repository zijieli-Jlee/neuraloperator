from functools import partialmethod

import torch.nn as nn
import torch.nn.functional as F

from ..layers.padding import DomainPadding
from ..layers.attention_kernel_integral import AttentionKernelIntegral
from ..layers.mlp import MLPLinear
from ..layers.embeddings import SirenNet, GaussianFourierFeatureTransform


class TransformerNO(nn.Module):
    """N-Dimensional Transformer-based Neural Operator (currently does not support N>2)
        using softmax-free attention to compute the kernel integral.

        Each layer in the encoder part is organized as follow:
            z = attn(norm(z)) + z
            z = mlp(norm(z)) + z
        where z is the input to the layer.

        For the decoder, given query bases q and latent embedding z:
            z = attn(q, z)

        Parameters
        ----------
        hidden_channels : int
            width of the model (i.e. number of channels)
        n_dim : int
            number of dimensions of the domain
        in_channels : int, optional
            Number of input channels, by default 1
        out_channels : int, optional
            Number of output channels, by default 1
        encoder_num_heads: int, optional
            Number of heads in the encoder attention, by default 1
        decoder_num_heads: int, optional
            Number of heads in the decoder cross-attention, by default 8
        encoder_dim_head: int, optional
            Dimension of the attention head in the encoder, by default equals to hidden_channels
        decoder_dim_head: int, optional
            Dimension of the attention head in the decoder, by default equals to hidden_channels
        query_basis: string, optional
            Type of query basis to use, by default 'siren', other options are ['fourier', 'linear']
        n_layers : int, optional
            Number of Transformer Layers in the encoder, by default 4
        use_mlp : bool, optional
            Whether to use an MLP layer after each attention block, by default True
        mlp_dropout : float , optional
            droupout parameter of MLP layer, by default 0
        mlp_expansion : float, optional
            expansion parameter of MLP layer, by default 2.0
        non_linearity : nn.Module, optional
            Non-Linearity module to use, by default F.gelu
        norm: string, optional
            Normalization module to use, by default layernorm

        """
    def __init__(self,
                 hidden_channels,
                 n_dim,
                 in_channels=1,
                 out_channels=1,
                 encoder_num_heads=1,
                 decoder_num_heads=8,
                 encoder_dim_head=None,
                 decoder_dim_head=None,
                 query_basis='siren',
                 n_layers=4,
                 use_mlp=True,
                 mlp_dropout=0,
                 mlp_expansion=2.0,
                 non_linearity=F.gelu,
                 norm='layer_norm',      # ['layer_norm', 'instance_norm', ''group_norm', 'none']
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_dim = n_dim
        self.encoder_num_heads = encoder_num_heads
        self.decoder_num_heads = decoder_num_heads
        self.encoder_dim_head = encoder_dim_head if encoder_dim_head is not None else hidden_channels
        self.decoder_dim_head = decoder_dim_head if decoder_dim_head is not None else hidden_channels
        self.query_basis = query_basis

        assert self.query_basis in ['siren', 'fourier', 'linear'],\
            f'query_basis must be one of ["siren", "fourier", "linear"], got {self.query_basis}'

        self.n_layers = n_layers
        self.use_mlp = use_mlp
        self.mlp_dropout = mlp_dropout
        self.mlp_expansion = mlp_expansion
        self.non_linearity = non_linearity
        self.norm = norm

        # top and bottom
        self.lifting = nn.Linear(self.in_channels, self.hidden_channels, bias=False)
        self.projection = MLPLinear([self.hidden_channels, self.hidden_channels, self.out_channels])

        # build encoder
        self.encoder_blocks = nn.ModuleList([])
        for layer in range(self.n_layers):
            encoder_layer = nn.ModuleList([])
            encoder_layer.append(self.get_normalization(self.norm, self.hidden_channels))
            encoder_layer.append(AttentionKernelIntegral(
                                                        dim=self.hidden_channels,
                                                        heads=self.encoder_num_heads,
                                                        dim_head=self.encoder_dim_head,
                                                        pos_dim=self.n_dim))
            if self.use_mlp:
                encoder_layer.append(self.get_normalization(self.norm, self.hidden_channels))
                encoder_layer.append(MLPLinear([self.hidden_channels,
                                                int(self.hidden_channels*mlp_expansion),
                                                self.hidden_channels], dropout=self.mlp_dropout))
            self.encoder_blocks.append(encoder_layer)

        # build decoder
        if self.query_basis == 'siren':
            self.query_basis_fn = SirenNet(dim_in=self.n_dim,
                                           dim_hidden=self.hidden_channels,
                                           dim_out=self.decoder_num_heads*self.decoder_dim_head,
                                           num_layers=3)
        elif self.query_basis == 'fourier':
            self.query_basis_fn = GaussianFourierFeatureTransform(self.n_dim,
                                                                  mapping_size=self.decoder_num_heads*self.decoder_dim_head//2,
                                                                  scale=2.0)
        elif self.query_basis == 'linear':
            self.query_basis_fn = nn.Linear(self.n_dim, self.decoder_num_heads*self.decoder_dim_head, bias=False)

        self.decoder = AttentionKernelIntegral(dim=self.hidden_channels,
                                                heads=self.decoder_num_heads,
                                                dim_head=self.decoder_dim_head,
                                                pos_dim=self.n_dim,
                                                project_query=False)
        self.decoder_norm = self.get_normalization(self.norm, self.hidden_channels)

    @staticmethod
    def get_normalization(norm, channels):
        if norm == 'none':
            norm_fn = nn.Identity()
        elif norm == "instance_norm":
            norm_fn = nn.InstanceNorm1d(channels)
        elif norm == "group_norm":
            norm_fn = nn.GroupNorm(num_groups=1, num_channels=channels)
        elif norm == 'layer_norm':
            norm_fn = nn.LayerNorm(channels)
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )
        return norm_fn

    def forward(self,
                x,
                input_pos,
                query_pos=None,
                **kwargs):
        """Transformer NO's forward pass,
           please note that coordinates must be normalized to [-1, 1] interval when using siren"""
        x = self.lifting(x)

        if self.use_mlp:
            for [norm_attn, attn, norm_ffn, ffn] in self.encoder_blocks:
                x = attn(norm_attn(x), input_pos) + x
                x = ffn(norm_ffn(x)) + x
        else:
            for [norm_attn, attn] in self.encoder_blocks:
                x = attn(norm_attn(x), input_pos) + x

        if query_pos is None:
            query_pos = input_pos
        z = x
        query_emb = self.query_basis_fn(query_pos)
        out = self.decoder(query_emb, query_pos, u_y=z, pos_y=input_pos)
        out = self.projection(self.decoder_norm(out))
        return out










