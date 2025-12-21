import re
import tiktoken


def file_loader(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("총 문자 개수: ", len(raw_text))

    return raw_text


def split_string(input_string):
    text = input_string
    splitted_text = re.split(r'([,.:;?)_!"()\']|--|\s)', text)
    splitted_text = [item.strip() for item in splitted_text if item.strip()]
    return splitted_text


def create_word_database(word_token):
    all_words = sorted(set(word_token))
    word_database = {word: i for i, word in enumerate(all_words)}
    return word_database


class Tokenizer:
    def __init__(self, word_db):
        self.str_to_int = word_db
        self.int_to_str = {i:s for s, i in word_db.items()}
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text):
        integers = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return integers
        
    def decode(self, integers):
        string = self.tokenizer.decode(integers)
        return string


def main():
    data_path = "./the_verdict.txt"
    string_data = file_loader(data_path)
    split_data = split_string(string_data)

    word_database = create_word_database(split_data)


if __name__ == "__main__":
    main()
