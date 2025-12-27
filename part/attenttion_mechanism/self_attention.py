import torch
import torch.nn as nn


SAMPLE_INPUT = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T
        attention_weight = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attention_weight @ values

        return context_vec


def main():
    torch.manual_seed(789)
    d_in = 3
    d_out = 2
    self_attention_module = SelfAttention(d_in, d_out)
    print(self_attention_module(SAMPLE_INPUT))


if __name__ == "__main__":
    main()
