import torch


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
