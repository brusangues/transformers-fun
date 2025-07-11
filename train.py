import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import json
from copy import deepcopy
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from src.data_loader import DataLoader, DataLoaderNgram
from src.gpt_v0 import GPTLanguageModel


nvmlInit()
torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device=}")


def main():
    parser = argparse.ArgumentParser(description="Train a GPT language model")
    parser.add_argument(
        "params", type=str, help="Name of the YAML params file in the params folder"
    )
    args = parser.parse_args()
    params = parse_yaml_params(f"params/{args.params}.yaml")
    training_loop(**params, device=device)


def parse_yaml_params(path):
    print(f"Loading parameters from {path}")
    with open(path, "r") as f:
        params = yaml.safe_load(f)
    print(f"Parameters loaded: {params}")
    return params


def training_loop(
    batch_size,
    context_len,
    max_iters,
    learning_rate,
    eval_interval,
    save_interval,
    eval_iters,
    n_embd,
    n_head,
    n_layer,
    dropout,
    device,
    path_input,
    path_load_model=None,
    path_save_model=None,
    n_tokens_generate=None,
    path_generate_output=None,
    name="",
    **kwargs,
):
    print("Starting training loop...")
    locals_ = deepcopy(locals())
    print(f"Parameters: {locals_}")

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/{name}/{now}")
    writer.add_text("params", str(locals_), 0)
    writer.flush()

    data_loader = DataLoaderNgram(context_len, batch_size, device)
    vocab_size, encode, decode = data_loader.load_data(path_input)

    def profile(iteration=0):
        info = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0))
        gpu_free = round(info.free / 1024**2, 2)
        gpu_used = round(info.used / 1024**2, 2)
        gpu_temp = torch.cuda.temperature()
        writer.add_scalar("gpu_free_memory", gpu_free, iteration)
        writer.add_scalar("gpu_used_memory", gpu_used, iteration)
        writer.add_scalar("gpu_temperature", gpu_temp, iteration)

    @torch.no_grad()
    def estimate_loss(iteration=0):
        print("Estimating loss...")
        out = {"iteration": iteration}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = data_loader.get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = float(losses.mean())
        model.train()
        print(
            f"step {iteration}: train loss {out['train']:.4f},"
            f" val loss {out['val']:.4f}"
        )
        return out

    @torch.no_grad()
    def generate_sample(n_tokens_generate=100):
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_text = decode(
            m.generate(context, max_new_tokens=n_tokens_generate)[0].tolist()
        )
        print("generated_text:", generated_text)
        return generated_text

    model = GPTLanguageModel(
        context_len=context_len,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        vocab_size=vocab_size,
        device=device,
    )

    m = model.to(device)
    if path_load_model:
        print(f"Loading model from {path_load_model}")
        m.load_state_dict(torch.load(path_load_model, map_location=device))
        print("Model loaded")
    # print the number of parameters in the model
    num_parameters = sum(p.numel() for p in m.parameters())
    print(num_parameters / 1e6, "M parameters")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    for iter in tqdm(range(max_iters), total=max_iters):
        # sample a batch of data
        xb, yb = data_loader.get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss_train", float(loss.mean()), iter)
        profile(iter)

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print()
            losses = estimate_loss(iter)
            writer.add_scalar("loss_train_estimated", losses["train"], iter)
            writer.add_scalar("loss_val", losses["val"], iter)
            generated_text = generate_sample()
            writer.add_text("generated_text", generated_text, iter)

        # SAVING
        if iter != 0 and (iter % save_interval == 0 or iter == max_iters - 1):
            # save model
            if path_save_model:
                checkpoint_name = f"{path_save_model}_{iter}.pth"
                print(f"Saving checkpoint to {checkpoint_name}")
                torch.save(m.state_dict(), checkpoint_name)
        writer.flush()

    print("Training finished. Generating text:")

    # generate from the model
    generated_text = generate_sample(n_tokens_generate)
    if path_generate_output:
        print(f"Saving generated text to {path_generate_output}")
        open(path_generate_output, "w").write(generated_text)
    print("Training loop finished.")
    writer.close()


if __name__ == "__main__":
    main()
