import torch
from torch import nn
import math
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn.functional as F
from einops import rearrange, repeat

from .mlp import MLPLinear
from .embeddings import RotaryEmbedding, apply_rotary_pos_emb, apply_2d_rotary_pos_emb


class AttentionKernelIntegral(torch.nn.Module):
    """
    Kernel integral transform with attention
    """

    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 pos_dim,
                 use_pe=True,    # use positional encoding
                 project_query=True
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.heads = heads
        self.dim_head = dim_head

        self.project_query = project_query
        if project_query:
            self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        else:
            self.to_q = nn.Identity()

        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)

        self.k_norm = nn.InstanceNorm1d(dim_head, affine=False)
        self.v_norm = nn.InstanceNorm1d(dim_head, affine=False)

        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)

        self.to_out = nn.Linear(dim_head * heads, dim) if dim_head * heads != dim else nn.Identity()

        self.use_pe = use_pe
        self.pos_dim = pos_dim

        if use_pe:
            self.pos_emb = RotaryEmbedding(dim_head // self.pos_dim, min_freq=1 / 64)

        self.init_gain = 1 / math.sqrt(dim_head)
        self.diagonal_weight = self.init_gain
        self.initialize_qkv_weights()

    def init_weight(self, weight, inif_fn):
        for param in weight.parameters():
            if param.ndim > 1:
                dim_head = param.size(0) // self.heads
                for h in range(self.heads):
                    inif_fn(param[h * dim_head:(h + 1) * dim_head, :], gain=self.init_gain)

                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                    torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def initialize_qkv_weights(self):
        init_fn = xavier_uniform_

        if self.project_query:
            self.init_weight(self.to_q, init_fn)
        self.init_weight(self.to_k, init_fn)
        self.init_weight(self.to_v, init_fn)

    def normalize_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self,
                u_x,
                pos_x,
                u_y=None,
                pos_y=None,
                weights=None,
                associate=True,   # can be faster if n is larger than the channel number c
                get_kernel=False):
        """
        Computes kernel integral transform with attention

        Parameters
        ----------
        u_x: input (query) function of shape [b, n, c]
        pos_x: coordinate of input function's grid points [b, n, d]
        u_y: the second source of function (key and value), if not provided, u_y = u_x
        pos_y: coordinate of the second source of function's grid points, if not provided, assume pos_y = pos_x
        weights : tensor of shape [b, n], if not provided assume to be 1/n
                Weights for each point y proprtional to the
                volume around f(y)=u_y W_v being integrated.
        associate: if True, use associativity of matrix multiplication, first multiply K^T V, then multiply Q
        get_kernel: if True, return the kernel matrix (for analyzing the kernel)
        Output
        ----------
        out_features: Output function given on the points x.
        """

        if u_y is None:
            u_y = u_x   # go back to self attention

        if get_kernel and associate:
            raise Exception('Cannot get attention when associate is True')

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)
        v = self.to_v(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        if weights is None:
            weights = torch.ones((u_y.shape[0], 1, u_y.shape[1], 1), device=u_y.device) / n
        else:
            weights = rearrange(weights, 'b n -> b () n ()')

        # q = self.q_norm(q)
        k = self.normalize_wrt_domain(k, self.k_norm)
        v = self.normalize_wrt_domain(v, self.v_norm)

        if self.use_pe:
            if self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_x = repeat(q_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_x = repeat(k_freqs_x, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, '1 n d -> b h n d', b=q.shape[0], h=k.shape[1])

                q = apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = self.pos_emb.forward(pos_x[..., 0], q.device).unsqueeze(0)
                q_freqs = repeat(q_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = self.pos_emb.forward(pos_y[..., 0], k.device).unsqueeze(0)
                    k_freqs = repeat(k_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding >= 3 dimensions')

        if associate:
            dots = torch.matmul(k.transpose(-1, -2), v)
            u = torch.matmul(q, dots) * weights
        else:
            # this is more efficient when n<<c
            kxy = torch.matmul(q, k.transpose(-1, -2))
            u = torch.matmul(kxy, v) * weights

        u = rearrange(u, 'b h n d -> b n (h d)')
        u = self.to_out(u)
        if get_kernel:
            return u, kxy
        return u

