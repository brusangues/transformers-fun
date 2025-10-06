import argparse
import yaml
import joblib
from copy import deepcopy
import random

from src.educational_tokenizer import SimpleBytePairEncoding


random.seed(1)


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "params", type=str, help="Name of the YAML params file in the params folder"
    )
    args = parser.parse_args()
    params = parse_yaml_params(f"params/{args.params}.yaml")
    train_tokenizer(**params)


def parse_yaml_params(path):
    print(f"Loading parameters from {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    print(f"Parameters loaded: {params}")
    return params


def train_tokenizer(
    path_input,
    path_tokenizer,
    max_tokens=5_000_000,
    vocab_size=1024,
    encode_all_bytes=False,
    visualise=10,
    test_string= "Não. Além. Caçada. Bônus gênero. — ª° ‡. í á â ^ à",
    **kwargs,
):
    print("Starting train_tokenizer loop...")
    locals_ = deepcopy(locals())
    print(f"Parameters: {locals_}")

    print("load_data...")
    with open(path_input, "r", encoding="utf-8") as f:
        text_full = f.read()
    
    chars = sorted(list(set(text_full)))
    print(f"{text_full[:2_000]=}")
    print(f"{len(text_full)=} {len(chars)=}\n{chars=}")

    print("Reduzindo dados para treinar o tokenizer...")
    start_idx = random.randint(0, max(0, len(text_full) - max_tokens - 1))
    print(f"Início aleatório: {start_idx=}")
    text = text_full[start_idx:start_idx + max_tokens]
    chars = sorted(list(set(text)))
    print(f"{text[:2_000]=}")
    print(f"{len(text)=} {len(chars)=}\n{chars=}")

    print("Treinando...")
    # Train a BPE tokenizer on a small amount of text
    GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    tokenizer = SimpleBytePairEncoding.train(text, vocab_size=vocab_size, pat_str=GPT4_PATTERN, visualise=visualise, encode_all_bytes=encode_all_bytes)

    print(f"Salvando tokenizer em {path_tokenizer}...")
    joblib.dump(tokenizer, path_tokenizer)

    print("Testando tokenizer...")
    tokens = tokenizer.encode(text, visualise=None)
    print(f"{len(text)=} {len(tokens)=}")
    compression_ratio = len(text) / len(tokens)
    print(f"{compression_ratio=}")
    # previous: len(text)=1523214 len(tokens)=712443 compression_ratio=2.13

    print("Testando codificação e decodificação...")
    tokens = tokenizer.encode(test_string)
    print(f"{test_string=}")
    print(f"{tokens=}")
    print(f"{tokenizer.decode(tokens)=}")
    print(tokenizer.decode(tokens) == test_string)

    print("Testando leitura do artefato...")
    tokenizer_ = joblib.load(path_tokenizer)
    print(f"{tokenizer_.decode(tokens)=}")
    print(tokenizer_.decode(tokens) == test_string)



if __name__ == "__main__":
    main()
