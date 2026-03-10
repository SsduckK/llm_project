import torch
import torch.nn as nn

from basemodule import BaseLayerNorm


class LayerNorm(BaseLayerNorm):
    def __init__(self, emb_dim):
        super().__init__(normalized_shape=emb_dim)
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("mean: ", mean)
    print("var: ", var)
