import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, file_loader, split_string, create_word_database


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


class LanguageDataLoader:
    def __init__(
        self,
        string_data,
        word_data_base,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        self.text_data = string_data
        self.word_database = word_data_base
        self.tokenizer = self.create_tokenizer(self.word_database)
        self.dataset = self.create_dataset(
            self.text_data, self.tokenizer, max_length, stride
        )
        self.dataloader = self.create_dataloader(
            self.dataset, batch_size, shuffle, drop_last, num_workers
        )

    def load_file(self, path):
        data = file_loader(path)
        return data

    def create_tokenizer(self, word_database):
        tokenizer = Tokenizer(word_database)

        return tokenizer

    def create_dataset(self, text, tokenizer, max_length, stride):
        string_data = split_string(text)
        dataset = GPTDataset(text, tokenizer, max_length, stride)
        return dataset

    def create_dataloader(self, dataset, batch_size, shuffle, drop_last, num_workers):
        print("dataset len =", len(dataset))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader

    def get_dataloader(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataloader)


def create_embedding_layer(word_database_size, context_size, output_dim):
    token_embedding_layer = torch.nn.Embedding(word_database_size, output_dim)
    position_embedding_layer = torch.nn.Embedding(context_size, output_dim)
    position_embedding = position_embedding_layer(torch.arange(context_size))
    return token_embedding_layer, position_embedding


def main():
    file_data = "./the_verdict.txt"
    string_data = file_loader(file_data)
    split_data = split_string(string_data)

    word_database = create_word_database(split_data)

    dataloader_class = LanguageDataLoader(string_data, word_database)
    dataloader = dataloader_class.get_dataloader()
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == "__main__":
    main()
