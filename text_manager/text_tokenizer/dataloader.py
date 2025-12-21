import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer, file_loader, split_string, create_word_database


class GPTDataset(Dataset):
    def __init__(self, text, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        self.token_id = self.create_token_id(text)
        self.matching_input_target_ids(self.token_id, max_length, stride)

    def create_token_id(self, text):
        string_data = file_loader(text)
        split_data = split_string(string_data)
        word_database = create_word_database(split_data)

        return word_database

    def matching_input_target_ids(self, token_ids, max_length, stride):
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]



