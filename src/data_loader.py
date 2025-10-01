import torch
import re
import joblib
import pandas as pd
from tqdm import tqdm
from .educational_tokenizer import SimpleBytePairEncoding

tqdm.pandas()


class DataLoader:

    def __init__(self, context_len, batch_size, device):
        self.context_len = context_len
        self.batch_size = batch_size
        self.device = device

    def load_data(self, path_input):
        print("load_data...")
        with open(path_input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"{text[:2_000]=}")

        # here are all the unique characters that occur in this text
        print(f"{len(text)=}")
        chars = sorted(list(set(text)))
        print(f"{chars=}")
        vocab_size = len(chars)
        print(f"{vocab_size=}")
        estimated_starting_loss = -torch.log(torch.ones(1) / vocab_size).item()
        print(f"{estimated_starting_loss=}")
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        # encoder: take a string, output a list of integers
        encode = lambda s: [stoi[c] for c in s]
        # decoder: take a list of integers, output a string
        decode = lambda l: "".join([itos[i] for i in l])

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.encode = encode
        self.decode = decode

        return vocab_size, encode, decode

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.context_len, (self.batch_size,))
        x = torch.stack([data[i : i + self.context_len] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_len + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class DataLoaderNgram:

    def __init__(self, context_len, batch_size, device, n_gram=2):
        self.context_len = context_len
        self.batch_size = batch_size
        self.device = device
        self.n_gram = n_gram

    def ngrams(self, word):
        n_odd_chars = len(word) % self.n_gram
        if n_odd_chars > 0:
            pad = self.n_gram - n_odd_chars
            word += " " * pad
        n_gram_list = [
            word[i : i + self.n_gram] for i in range(0, len(word), self.n_gram)
        ]
        return n_gram_list

    def tokenize(self, text_sample):
        new_string = re.sub("(\s)", "\\1[cut]", text_sample)
        words = new_string.split("[cut]")
        # words = text_sample.split()
        words = [w for w in words if w != "" and w != " "]
        words = [re.sub(" +$", " ", re.sub("$", " ", w)) for w in words]
        # print(words)
        n_grams_list = []
        for w in words:
            n_grams_list.extend(self.ngrams(w))
        n_grams_list
        return n_grams_list

    def detokenize(self, n_grams_list):
        text = "".join(n_grams_list)
        return re.sub(" +", " ", text)

    def load_data(self, path_input):
        print("load_data...")
        with open(path_input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"{text[:2_000]=}")
        print(f"{len(text)=}")

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        print(f"{len(chars)=} {chars=}")

        tokens_full = self.tokenize(text)
        print(f"{len(tokens_full)=} {tokens_full[:300]=}")

        tokens = sorted(list(set(tokens_full)))
        print(f"{tokens[:300]=}")
        vocab_size = len(tokens)
        print(f"{vocab_size=}")
        estimated_starting_loss = -torch.log(torch.ones(1) / vocab_size).item()
        print(f"{estimated_starting_loss=}")
        # create a mapping from characters to integers
        stoi = {t: i for i, t in enumerate(tokens)}
        itos = {i: t for i, t in enumerate(tokens)}
        # encoder: take a string, output a list of integers
        encode = lambda s: [stoi[c] for c in self.tokenize(s)]
        # decoder: take a list of integers, output a string
        decode = lambda l: self.detokenize([itos[i] for i in l])
        assert decode(encode("ola mundo ")) == "ola mundo "

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.encode = encode
        self.decode = decode

        return vocab_size, encode, decode

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.context_len, (self.batch_size,))
        x = torch.stack([data[i : i + self.context_len] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_len + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class DataLoaderSimpleBpe:

    def __init__(
        self,
        context_len,
        batch_size,
        device,
        path_tokenizer="artifacts/tokenizers/tokenizer_simple_bpe_v2.joblib",
    ):
        self.context_len = context_len
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = joblib.load(path_tokenizer)

    def load_data(self, path_input):
        print("load_data...")
        with open(path_input, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"{text[:2_000]=}")
        print(f"{len(text)=}")

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        print(f"{len(chars)=} {chars=}")

        tokens_full = self.tokenizer.encode(text, visualise=None)
        print(f"{len(tokens_full)=} {tokens_full[:300]=}")

        tokens = sorted(list(set(tokens_full)))
        print(f"{tokens[:300]=}")

        vocab_size_tokens = len(tokens)
        print(f"{vocab_size_tokens=}")
        estimated_starting_loss = -torch.log(torch.ones(1) / vocab_size_tokens).item()
        print(f"{estimated_starting_loss=}")

        vocab_size_tokenizer = len(self.tokenizer.mergeable_ranks)
        print(f"{vocab_size_tokenizer=}")
        estimated_starting_loss_ = -torch.log(torch.ones(1) / vocab_size_tokenizer).item()
        print(f"{estimated_starting_loss_=}")

        # encoder: take a string, output a list of integers
        encode = lambda s: self.tokenizer.encode(s, visualise=None)
        # decoder: take a list of integers, output a string
        decode = lambda l: self.tokenizer.decode(l)
        assert decode(encode("ola mundo ")) == "ola mundo "

        # Train and test splits
        data = torch.tensor(encode(text), dtype=torch.long)
        n = int(0.9 * len(data))  # first 90% will be train, rest val
        train_data = data[:n]
        val_data = data[n:]

        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size_tokenizer
        self.encode = encode
        self.decode = decode

        return vocab_size_tokenizer, encode, decode

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.context_len, (self.batch_size,))
        x = torch.stack([data[i : i + self.context_len] for i in ix])
        y = torch.stack([data[i + 1 : i + self.context_len + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y


class DataLoaderBpeV2:

    def __init__(
        self,
        context_len,
        batch_size,
        device,
        path_tokenizer="artifacts/tokenizers/mix_ptbr_4096_v1.joblib",
        path_input = "data/df_full_v0.pq",
        texts_sample_frac = 1.0,
        split_frac = 0.8,
    ):
        print("DataLoaderBpeV2 init...")
        self.context_len = context_len
        self.batch_size = batch_size
        self.device = device
        self.split_frac = split_frac
        print(f"{context_len=}, {batch_size=}, {device=}, {split_frac=}")
        self.tokenizer = joblib.load(path_tokenizer)
        self.vocab_size_tokenizer = len(self.tokenizer.mergeable_ranks)
        print(f"{self.vocab_size_tokenizer=}")
        self.df_full = pd.read_parquet(path_input).query("train").sample(frac=texts_sample_frac, random_state=1).reset_index(drop=True)
        print(f"{self.df_full.shape=}")
        print(f"{self.df_full.value_counts(['author']).sort_index().reset_index()=}")
        print(f"{self.df_full.value_counts(['author','class']).sort_index().reset_index()=}")

    def load_data(self):
        print("load_data...")
        
        print(f"{self.df_full.shape}")
        text = "\n\n\n".join(self.df_full.text_clean.to_list())
        text = text[:1_000_000]
        print(f"{text[:2_000]=}")
        print(f"{len(text)=}")

        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        print(f"{len(chars)=} {chars=}")

        tokens_full = self.tokenizer.encode(text, visualise=None)
        print(f"{len(tokens_full)=} {tokens_full[:300]=}")

        tokens = sorted(list(set(tokens_full)))
        print(f"{tokens[:200]=}")

        vocab_size_tokens = len(tokens)
        print(f"{vocab_size_tokens=}")
        estimated_starting_loss = -torch.log(torch.ones(1) / vocab_size_tokens).item()
        print(f"{estimated_starting_loss=}")

        vocab_size_tokenizer = len(self.tokenizer.mergeable_ranks)
        print(f"{vocab_size_tokenizer=}")
        estimated_starting_loss_ = -torch.log(torch.ones(1) / vocab_size_tokenizer).item()
        print(f"{estimated_starting_loss_=}")

        # encoder: take a string, output a list of integers
        encode = lambda s: self.tokenizer.encode(s, visualise=None)
        # decoder: take a list of integers, output a string
        decode = lambda l: self.tokenizer.decode(l)
        assert decode(encode("ola mundo ")) == "ola mundo "

        # Train and test splits
        print("Encoding texts...")
        self.df_full["text_encoded"] = self.df_full.text_clean.progress_apply(lambda x: torch.tensor(encode(x), dtype=torch.long))
        print(f"{self.df_full.text_encoded.apply(len).describe()=}")
        index_train = self.df_full.sample(frac=self.split_frac, random_state=1).index
        self.df_full["split"] = pd.Series(self.df_full.index.isin(index_train)).map({True: "train", False: "eval"})
        print(f"{self.df_full.value_counts(['split','author']).sort_index().reset_index()=}")
        print(f"{self.df_full.value_counts(['split','author','class']).sort_index().reset_index()=}")

        self.encode = encode
        self.decode = decode

        return vocab_size_tokenizer, encode, decode

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        df_split = self.df_full.query("split=='train'") if split == "train" else self.df_full.query("split=='eval'")
        df_batch = df_split.sample(self.batch_size, replace=True)

        x = torch.zeros((self.batch_size, self.context_len), dtype=torch.long)
        y = torch.zeros((self.batch_size, self.context_len), dtype=torch.long)
        for i, (id_row, row) in enumerate(df_batch.iterrows()):
            # print(f"{i=} {id_row=} {row.author=} {row['class']=}{len(row.text_encoded)=}")
            t = row.text_encoded
            if len(t) <= self.context_len:
                t = torch.cat([t, torch.zeros(self.context_len, dtype=torch.long)])
                start_idx = 0
            else:
                start_idx = torch.randint(0, len(t) - self.context_len, (1,)).item()
            # print(f"{len(t)=} {start_idx=} {start_idx + self.context_len=}")
            x[i] = t[start_idx : start_idx + self.context_len]
            y[i] = t[start_idx + 1 : start_idx + self.context_len + 1]

        x, y = x.to(self.device), y.to(self.device)
        return x, y
