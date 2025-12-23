### text tokenize


신경망은 텍스트 자체를 직접 처리할 수는 없다. 연속적인 수치 연산을 기반으로 동작하기 때문에 텍스트를 그대로 연산 대상으로 사용할 수 없다. 따라서 문자열 형태의 텍스트 데이터를 정수나 실수 기반으로 변환해야 한다. 이를 위해 텍스트 데이터를 단어, 문자 기반의 개별 토큰으로 나눠야 한다. 나눈 토큰들은 임베딩을 통해 수치 벡터로 변환하여 신경망에 투입 가능하게 된다.

토큰 ID는 각 토큰(단어 혹은 subword)에 부여된 **정수 인덱스**이며, Python 환경에서 vocabulary(dict) 형태로 쉽게 구현할 수 있다. 단어 데이터베이스를 구축하고, 그 데이터베이스로부터 다음 토큰을 예측하는 입력–출력 관계를 정의하는 것부터 시작한다. 이러한 과정에서 특수 토큰(`<UNK>`, `<PAD>`, `<BOS>`, `<EOS>` 등)을 추가하여  모델이 **학습되지 않은 입력**, **문장 경계**, **패딩 영역** 등을 명시적으로 인식할 수 있도록 한다.

단어 단위 토큰화 방식은 vocabulary에 없는 단어(OOV)를 처리하기 어렵기 때문에,  
GPT-2, GPT-3에서는 **subword 기반 토큰화 방식인 BPE(Byte Pair Encoding)** 를 사용한다.

[[Byte Pair Encoding]] - 추가 예정

Sliding window 방식을 이용해서 입력 - 출력 간의 쌍을 생성
입력 시퀀스가 [1, 2, 3, 4, 5] 라 하면 sliding window 방식으로 아래와 같은 입력 - 출력 쌍을 생성할 수 있다. 
``1 - 2 ``
``2 - 3 ``
``3 - 4 ``
``4 - 5``
이는 **다음 토큰 예측(next-token prediction)** 을 위한 학습 데이터 구성 방식이다.

Pytorch Embedding Layer 를 통해서 입력 토큰 ID에 해당하는 벡터를 생성하도록 한다. 이 임베딩 레이어를 통해서 특정 단어에 해당하는 토큰 - 그에 맞는 벡터를 뽑아낼 수 있도록 하고 이는 이후 신경망 모델의 입력으로 사용된다.

토큰 임베딩을 통해 각 토큰은 고정 길이의 벡터로 변환된다. 이 임베딩 벡터에는 토큰의 순서 정보가 포함되지 않는데, 위치 정보가 없을 경우 모델에서는 순서에 상관없이 입력을 받기 때문에 어순을 살릴 수 없게 된다. self-attention 연산 자체가 순서 정보를 내재하지 않기 때문이다.
따라서 위치 정보를 나타내는 값을 추가하지 않을 경우 모델은 입력 시퀀스의 순서를 구분할 수 없게 된다.
예시를 들면 "I am student." 라는 문자열이 입력될 때 이를 각각 토큰화 하면 "I", "am", "student", "." 총 4개의 토큰이 나오게 되는데 24개(4!) 만큼의 순서 조합이 나올 수 있지만 위치 정보가 없을 경우 이 모두가 모델에서는 같은 의미로 해석이 되게 되는 문제가 있다.
즉 "I student am.", "I . am student", "I . am student"... 을 비롯한 모든 텍스트가 같은 의미로 파악되는 문제가 있다. 이를 해결하기 위해 위치 임베딩 정보를 추가하여 어순에 대한 각 토큰에 대해 상대적, 절대적 위치 정보를 명시적으로 제공한다. 이는 추후 attention 에서 더 자세히 다룰 예정


---

### 코드 정리
- Eidth Wharton 의 단편 소설 The Verdict(심판) 을 샘플 데이터로 사용
    - [https://en.wikisource.org/wiki/The_Verdict](https://en.wikisource.org/wiki/The_Verdict)
**tokenizer.py**
```python
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
```
- file_loader
텍스트 파일 경로를 받아서 총 문자 수를 출력하고, 로드한 파일을 반환
- split_string
입력 문자열을 정규 표현식으로 나눈 이후 공백 제외 list 로 만들어서 반환
- create_word_database
문자 토큰들을 받아서 중복 제거와 오름차순으로 정렬 이후 enumerate를 통해 인덱스와 토큰들을 매칭한다.
인덱스는 토큰 ID 에 해당

```Python
class Tokenizer:
    def __init__(self, word_db, model="gpt2"):
        self.str_to_int = word_db
        self.int_to_str = {i: s for s, i in word_db.items()}
        self.tokenizer = tiktoken.get_encoding(model)

    def encode(self, text):
        integers = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return integers

    def decode(self, integers):
        string = self.tokenizer.decode(integers)
        return string

    def __len__(self):
        return self.tokenizer.n_vocab
```
- Tokenizer 클래스
tiktoken 과 python dict 를 사용한 vocabulary 방식이 혼용되어서 제대로 정리되지 않은 상태
str_to_int, int_to_str 는 word_db 를 받아서 각각 인코딩하거나 디코딩 하는데 사용되지만 현재 코드에는 구현되어있지 않는 상태이다.
tokenizer 는 tiktoken 모듈을 이용하며 별도 지정이 없을경우 gpt2 모델을 사용한다.
encode 는 문자열을 수로 인코딩
decode 는 수를 문자로 디코딩
______len_____ 은 tokenizer 에 들어간 총 단어 수. gpt2 모델의 경우 50257 개이다.


**dataloader.py**
```Python
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
        for i in range(0, len(token_ids) - max_length - 1, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
```
- GPTDataset
Pytorch Dataset 모듈 상속 받아 사용한다.
입력 - 출력 순서 쌍을 만들기 위해서 self.input_ids, target_ids 를 준비하였으며 이들은 비어있는 상태로 시작. making_input_target_ids 에서 채워질 예정이다.
사용할 tokenizer 로부터 입력 텍스트에 대한 token id를 create_token_id  함수를 통해 생성. 이를 self.token_id 에 저장한다.
self.matching_input_target_ids 함수에서 token_id, max_length, stride 를 통해 입력 - 출력 순서쌍을 채운다.
해당 함수는 sliding window 방식을 사용
![[Pasted image 20251224015847.png]]
각 행은 길이 `max_length`의 입력 시퀀스와, 한 토큰 오른쪽으로 shift된 출력 시퀀스 쌍을 나타낸다.  
stride=1인 sliding window 방식으로 학습 데이터를 생성한다.

Dataset 의 길이를 알 수 있는 _____len_____ 함수와 인덱스를 통해서 입력 - 출력 순서쌍을 동시에 가져올 수 있는 _____getitem_____ 함수


```Python
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

    def get_word_db(self):
        return self.tokenizer

```
- LanguagetDataLoader 클래스
앞서 만들었던 GPTDataset 과 Tokenizer 클래스를 받아서 사용할 예정이다.
데이터 로더 생성 시점에서 로드된 text 데이터와 word_data_base 를 받는다. 여기서 문제가 있는데, 아까 Tokenizer 가 python 기반의 dictionary word database 를 구성한 것과 tiktoken 의 구조가 섞여있던 거 때문에 해당 데이터 로더 설계 시 불필요한 word_data_base 가 들어가야만 하게 되었다. 현재 파트에서는 각각 토크나이저와 데이터로더 생성을 연습하는 파트이므로 일단 이대로 두고, 추후 본격적으로 모델 구성할 때 이런 실수를 반복하지 않도록 해야 한다.
이후 batch, max_length(window_size), stride, shuffle, dropt_last, num_workers 등의 데이터 로더에서 필수적으로 필요한 요소를 기본적으로 제공해준 이후, 필요한 데이터들을 각각 self로 등록한다.
여기에는 데이터 로더에서 사용할 텍스트 데이터, 단어 데이터베이스(tiktok 을 사용하면 필요 없지만 현재 사용할 Tokenizer 클래스의 구현상 들어가야만 한다.), tokenizer(tiktok - gpt2 모델을 이용한), dataset 등을 생성해서 데이터 사용 준비를 마친다.


```Python
def create_embedding_layer(word_database_size, context_size, output_dim):
    token_embedding_layer = torch.nn.Embedding(word_database_size, output_dim)
    position_embedding_layer = torch.nn.Embedding(context_size, output_dim)
    position_embedding = position_embedding_layer(torch.arange(context_size))
    return token_embedding_layer, position_embedding
```
- create_embedding_layer
각 단어들을 특정 사이즈로 임베딩 해주는 역할을 하여, 이를 token_embedding_layer 에 등록한다.
`position_embedding_layer = torch.nn.Embedding(context_size, output_dim)`은 시퀀스의 각 위치 인덱스(0 ~ context_size-1)에 대해 학습 가능한 위치 임베딩 벡터를 제공한다. `torch.arange(context_size)`로 위치 인덱스 텐서를 만든 뒤 이를 `position_embedding_layer`에 통과시키면 `(context_size, output_dim)` 형태의 위치 임베딩 행렬이 생성된다. 이 위치 임베딩은 토큰 임베딩과 동일한 차원(`output_dim`)을 가지므로, `token_embedding + position_embedding`처럼 더해 self-attention에 입력함으로써 모델이 어순(순서) 정보를 활용할 수 있게 된다. **포지션 임베딩은 단어 사전 크기(vocab_size)가 아니라 시퀀스 길이(context_size)에 의해 결정된다.**
이렇게 생성된 position_embedding_layer 에 context_size 크기의 0부터 시작하는 torch tensor 를 넣으므로 context_size(max_length) 범위의 인덱스에 해당하는 포지션 벡터가 준비된다.
이렇게 생성된 임베딩 정보 두 개를 반환한다.
![[Pasted image 20251224022600.png]]
![[Pasted image 20251224022607.png]]

```python
def main():
    file_data = "./the_verdict.txt"
    max_length = 4
    output_dim = 4
    string_data = file_loader(file_data)
    split_data = split_string(string_data)

    word_database = create_word_database(split_data)

    dataloader_class = LanguageDataLoader(
        string_data, word_database, max_length=max_length
    )
    dataloader = dataloader_class.get_dataloader()
    word_db = dataloader_class.get_word_db()
    data_iter = iter(dataloader)
    inputs, target = next(data_iter)

    print(inputs)
    print(target)
    print("vocab_size =", len(word_db))
    print("inputs min/max =", inputs.min().item(), inputs.max().item())

    token_embedding_layer, position_embedding = create_embedding_layer(
        len(word_db), max_length, output_dim=output_dim
    )

    token_embedding = token_embedding_layer(inputs)

    input_embedding = token_embedding + position_embedding
    print(input_embedding)
    print(input_embedding.shape)
```

