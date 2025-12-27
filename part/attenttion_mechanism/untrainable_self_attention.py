import torch

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


def get_attention_score_single():
    query = SAMPLE_INPUT[1]
    attention_scores = torch.empty(SAMPLE_INPUT.shape[0])

    for i, x_i in enumerate(SAMPLE_INPUT):
        attention_scores[i] = torch.dot(x_i, query)

    return attention_scores


def get_attention_scores():
    attn_scores = torch.empty(6, 6)

    for i, x_i in enumerate(SAMPLE_INPUT):
        for j, x_j in enumerate(SAMPLE_INPUT):
            attn_scores[i, j] = x_i @ x_j

    return attn_scores


def normalize_tensor(input_tensor):
    val = torch.sum(input_tensor, dim=0)

    return input_tensor / val


def main():
    atteion_score_single = get_attention_score_single()
    atteion_scores = get_attention_scores()
    print("second row:", atteion_score_single)
    print("total:", atteion_scores)

    normalize_tensor(atteion_scores)


if __name__ == "__main__":
    main()
