import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, file_loader, split_string, create_word_database


def create_dataloader(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    string_data = file_loader(text)
    split_data = split_string(string_data)
    word_database = create_word_database(split_data)

    tokenizer = Tokenizer(word_database)

    dataset = GPTDataset(string_data, tokenizer, max_length, stride)

    print("dataset len =", len(dataset))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        self.token_id = self.create_token_id(text, tokenizer)
        self.matching_input_target_ids(self.token_id, max_length, stride)

    def create_token_id(self, text, tokenizer):
        token_ids = tokenizer.encode(text)
        return token_ids

    def matching_input_target_ids(self, token_ids, max_length, stride):
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def main():
    file_data = "./the_verdict.txt"
    dataloader = create_dataloader(
        file_data, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == "__main__":
    main()
