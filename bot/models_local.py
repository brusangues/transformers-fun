import torch
import yaml
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from transformers import AutoTokenizer

from src.gpt_v0 import GPTLanguageModel


nvmlInit()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{DEVICE=}")

LOCAL_MODELS = {
    "v22": ("models/v22_kaggle_0/params.yaml", "models/v22_kaggle_0/checkpoints/best_15000.pth"),
    "v22_test": ("models/v22_test/params.yaml", "models/v22_test/checkpoints/1999.pth"),
    "v23": ("models/v23_kaggle_0/params.yaml", "models/v23_kaggle_0/checkpoints/best_16000.pth"),
}


def profile():
    info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
    gpu_free = round(info.free / 1024**2, 2)
    gpu_used = round(info.used / 1024**2, 2)
    gpu_temp = torch.cuda.temperature()
    print("gpu/free_memory", gpu_free)
    print("gpu/used_memory", gpu_used)
    print("gpu/temperature", gpu_temp)


def parse_yaml_params(path):
    print(f"Loading parameters from {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    print(f"Parameters loaded: {params}")
    return params


def load_local_model(model_name="v22"):
    params_path, model_path = LOCAL_MODELS[model_name]
    params = parse_yaml_params(params_path)
    tokenizer, model = load_model(
        path_load_model = model_path,
        **params,
    )
    generate_sample(tokenizer, model, "Oi", 10)
    return tokenizer, model


def load_model(
    context_len,
    n_embd,
    n_feed_forward,
    n_head,
    n_layer,
    dropout,
    path_tokenizer,
    path_load_model,
    device=DEVICE,
    **kwargs,
):
    print("load_local_model")
    profile()

    print("Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path_tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"{tokenizer=}")

    print("\nInitializing model...")
    model = GPTLanguageModel(
        context_len=context_len,
        n_embd=n_embd,
        n_feed_forward=n_feed_forward,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        vocab_size=vocab_size,
        device=device,
    )

    m = model.to(device)
    profile()
    print(f"Loading model from {path_load_model}")
    m.load_state_dict(torch.load(path_load_model, map_location=device))
    print("Model loaded")
    print(f"{model=}")
    # print the number of parameters in the model
    n_parameters = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"{n_parameters=} M parameters")

    return tokenizer, model


@torch.no_grad()
def generate_sample(tokenizer, model, context: str = "", n_tokens_generate=100, top_k=30, temperature=0.7):
    encode = lambda s: tokenizer(s, truncation=False).input_ids
    decode = lambda l: tokenizer.decode(l)
    print(f"Generating {n_tokens_generate} tokens from {context=}")
    if context:
        context = torch.tensor(encode(context), dtype=torch.long, device=DEVICE).unsqueeze(0)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_text = decode(
        model.generate(context, max_new_tokens=n_tokens_generate, top_k=top_k, temperature=temperature)[0].tolist()
    )
    profile()
    print("generated_text:", generated_text)
    return generated_text


if __name__ == "__main__":
    load_local_model("v22")
