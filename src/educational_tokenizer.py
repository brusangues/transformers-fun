"""This is an educational implementation of the byte pair encoding algorithm."""

import collections

import regex

import tiktoken
from tqdm import tqdm

replacement_char = "�"


class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        print(f"{len(mergeable_ranks)=} {pat_str=}")
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {
            token: token_bytes for token_bytes, token in mergeable_ranks.items()
        }
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: str | None = "colour") -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(
                self.mergeable_ranks, word_bytes, visualise=visualise
            )
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str, visualise: str, encode_all_bytes: bool = True):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(
            data=training_data,
            vocab_size=vocab_size,
            pat_str=pat_str,
            visualise=visualise,
            encode_all_bytes=encode_all_bytes,
        )
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = "colour"
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise:
            if visualise in ["colour", "color"]:
                visualise_tokens(parts)
            elif visualise == "simple":
                print(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )

    if visualise:
        print()

    # Unk token added
    tokens = [mergeable_ranks.get(part, mergeable_ranks[bytes(replacement_char.encode("utf-8"))]) for part in parts]
    return tokens


def bpe_train(
    data: str, vocab_size: int, pat_str: str, visualise: str | None = "colour", encode_all_bytes: bool = True
) -> dict[bytes, int]:
    # Splinter up our data into lists of bytes
    # data = "Hello world"
    # words = [
    #     [b'H', b'e', b'l', b'l', b'o'],
    #     [b' ', b'w', b'o', b'r', b'l', b'd']
    # ]
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")]
        for word in regex.findall(pat_str, data)
    ]
    print(f"Initial 20 words: {words[:20]}")

    ranks = {}
    if encode_all_bytes:
        print("First, add tokens for each individual byte value")
        for i in range(2**8):
            ranks[bytes([i])] = i
        print(f"Added all {2**8} byte values")
    else:
        print("Add the individual characters that actually occur in the data")
        pieces = [bytes([b]) for b in data.encode("utf-8")]
        chars = sorted(list(set(pieces)))
        print(f"{len(chars)=} {chars=}")
        added_chars = []
        for i, c in enumerate(chars):
            # print(f"{i=} {c=} {bytes(c)=} {len(bytes(c))=}:", end=" ")
            if (bytes(c) not in ranks) and len(bytes(c))==1:
                # print(f"added.", end=" ")
                ranks[bytes(c)] = len(ranks)
                added_chars.append(c)
        print(f"\nAdded {len(added_chars)} characters: {added_chars}")

    # Adding unknown token
    unknown_token = len(ranks)
    print("Adding unknown token", unknown_token)
    ranks[bytes(replacement_char.encode("utf-8"))] = unknown_token

    ranks_readable = [(v, k, k.decode("utf-8", errors="replace")) for k, v in ranks.items()]
    print(f"Initial ranks: {ranks_readable}")

    # Now, use our data to figure out which merges we should make

    # while len(ranks) < vocab_size:
    # tqdm
    total = vocab_size - len(ranks)
    for it in tqdm(range(total), desc="Training BPE", unit=" merges", ncols=100):
        # Find the most common pair. This will become our next token
        stats = collections.Counter()
        for piece in words:
            for pair in zip(piece[:-1], piece[1:]):
                stats[pair] += 1

        most_common_pair = max(stats, key=lambda x: stats[x])
        token_bytes = most_common_pair[0] + most_common_pair[1]
        token = len(ranks)
        # token_decoded = token_bytes.decode("utf-8", errors="replace")
        # print(f"{token=} {token_bytes=} {token_decoded=}")
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == most_common_pair:
                    # We found our pair! Merge it
                    new_word.append(token_bytes)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            if i == len(word) - 1:
                new_word.append(word[i])
            new_words.append(new_word)
        words = new_words

        # See the intermediate merges play out!
        if visualise == "simple":
            print(
                f"{it} - Current most common pair: {most_common_pair[0]} + {most_common_pair[1]}"
                f" -> {token_bytes} = {len(ranks)}th token"
            )
            for word in words[10:50]:
                print(word)
            print()
        elif visualise and it % visualise == 0:
            print(
                f"{it} - Current most common pair: {most_common_pair[0]} + {most_common_pair[1]}"
                f" -> {token_bytes} = {len(ranks)}th token"
            )
            visualise_tokens([token for word in words[10:50] for token in word])
            print()

    ranks_readable = [(v, k, k.decode("utf-8", errors="replace")) for k, v in ranks.items()]
    print(f"Final ranks: {ranks_readable}")

    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def train_simple_encoding():
    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    with open(__file__) as f:
        data = f.read()

    enc = SimpleBytePairEncoding.train(data, vocab_size=600, pat_str=gpt2_pattern)

    print("This is the sequence of merges performed in order to encode 'hello world':")
    tokens = enc.encode("hello world")
    assert enc.decode(tokens) == "hello world"
    assert enc.decode_bytes(tokens) == b"hello world"
    assert enc.decode_tokens_bytes(tokens) == [b"hello", b" world"]

    return enc
