from einops import rearrange, repeat

import torch  # noqa
from torch import nn

from neural_gfca.nn import TACModule


class RESepFormerBlock(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()

        def tf_generator():
            return nn.TransformerEncoderLayer(
                d_model,
                n_head,
                dim_feedforward,
                batch_first=True,
                norm_first=norm_first,
                layer_norm_eps=layer_norm_eps,
            )

        self.intra_tf_list = nn.ModuleList([tf_generator() for _ in range(n_layers)])
        self.inter_tf_list = nn.ModuleList([tf_generator() for _ in range(n_layers)])

        self.tac_list = nn.ModuleList([TACModule(d_model) for _ in range(n_layers)])

        self.last_intra_tf = tf_generator()
        self.gln = nn.GroupNorm(1, d_model)

    def forward(self, x: torch.Tensor, n_mic: int):
        B, S, T, C = x.shape

        h = rearrange(x, "b s t c -> (b s) t c")
        for intra_tf, inter_tf, tac in zip(self.intra_tf_list, self.inter_tf_list, self.tac_list, strict=False):
            # intra-chunk
            h = intra_tf(h)

            # inter-mic (tac)
            h = rearrange(tac(rearrange(h, "(b s) t c -> b (s t) c", s=S), n_mic), "b (s t) c -> (b s) t c", s=S)

            # inter-chunk
            h = h + repeat(inter_tf(rearrange(h.mean(dim=1), "(b s) c -> b s c", s=S)), "b s c -> (b s) t c", t=T)

        # intra-chunk
        h = rearrange(self.last_intra_tf(h), "(b s) t c -> b s t c", s=S)
        h = rearrange(self.gln(rearrange(h, "b s t c -> b c (s t)")), "b c (s t) -> b s t c", s=S)

        return h
