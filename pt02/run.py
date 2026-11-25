import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pt01.run import BigramModel, encode_word, load_words


def encode_words(words):
    result = []

    for word in words:
        if word not in word_id_cache:
            tokens = encode_word(word, ranks)
            word_id_cache[word] = [token_to_id[token] for token in tokens]

        result.extend(word_id_cache[word])

    return result


def sample_windows(token_ids, batch_size, context_size, rng):
    starts = torch.randint(
        0, len(token_ids) - context_size, (batch_size,), generator=rng
    )

    windows = []
    for start in starts.tolist():
        window = token_ids[start : start + context_size + 1]
        windows.append(window)

    windows = torch.stack(windows)
    return windows[:, :-1], windows[:, 1:]


class CausalAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)

    def split_heads(self, tensor):
        batch_size, length, _ = tensor.shape
        return tensor.reshape(
            batch_size, length, self.num_heads, self.head_size
        ).transpose(1, 2)

    def forward(self, x):
        batch_size, length, width = x.shape

        queries = self.split_heads(self.query(x))
        keys = self.split_heads(self.key(x))
        values = self.split_heads(self.value(x))

        scores = queries @ keys.transpose(-2, -1) / self.head_size**0.5
        future_mask = torch.ones(
            length, length, dtype=torch.bool, device=x.device
        ).triu(diagonal=1)

        weights = torch.softmax(
            scores.masked_fill(future_mask, float("-inf")), dim=-1
        )
        attended = weights @ values

        combined = attended.transpose(1, 2).reshape(batch_size, length, width)
        return self.output(combined)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalAttention(d_model, num_heads)
        self.ff_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.ff_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, context_size):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(context_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(layer):
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            nn.init.normal_(layer.weight, std=0.02)
            if isinstance(layer, nn.Linear) and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, token_ids):
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x)

        return self.output(self.final_norm(x))


def next_token_loss(model, inputs, targets):
    logits = model(inputs)
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )


@torch.no_grad()
def generate(model, prompt, max_new_tokens, temperature=1.0, top_p=None):
    prompt_words = re.findall(
        r"[а-я]+(?:-[а-я]+)*", prompt.lower().replace("ё", "е")
    )
    ids = encode_words(prompt_words)
    new_ids = []
    device = next(model.parameters()).device
    rng = torch.Generator().manual_seed(42)

    for _ in range(max_new_tokens):
        context = torch.tensor([ids[-model.context_size:]], device=device)
        logits = model(context)[0, -1].float().cpu()

        if temperature == 0:
            next_id = logits.argmax().item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)

            if top_p is None:
                next_id = torch.multinomial(probs, 1, generator=rng).item()
            else:
                sorted_probs, order = probs.sort(descending=True)
                cumulative = sorted_probs.cumsum(dim=-1)
                keep = torch.cat((torch.tensor([True]), cumulative[:-1] < top_p))
                sampled = torch.multinomial(sorted_probs * keep, 1, generator=rng)
                next_id = order[sampled].item()

        ids.append(next_id)
        new_ids.append(next_id)

    text = "".join(vocabulary[i] for i in ids).replace("</w>", " ").strip()
    return text, new_ids


@torch.no_grad()
def measure_loss(model, inputs, targets):
    total = 0.0
    for x, y in zip(inputs.split(4), targets.split(4)):
        total += next_token_loss(model, x, y).item() * len(x)
    return total / len(inputs)


def train_model(train_ids, val_ids, vocab_size, fast=False):
    steps = 200 if fast else 2000
    batch_size = 16 if fast else 32
    interval = 25 if fast else 100
    sample_steps = {0, 50, 100, steps} if fast else {0, 100, 500, steps}
    context_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    torch.manual_seed(42)
    model = TinyTransformer(vocab_size, 128, 4, 512, 4, context_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train_rng = torch.Generator().manual_seed(42)
    check_rng = torch.Generator().manual_seed(43)

    x_start, y_start = sample_windows(train_ids, 20, context_size, check_rng)
    x_val, y_val = sample_windows(val_ids, 20, context_size, check_rng)
    x_start, y_start = x_start.to(device), y_start.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    curve = []
    samples = []

    def record(step, train_loss):
        model.eval()
        val_loss = measure_loss(model, x_val, y_val)
        curve.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})

        if step in sample_steps:
            text, _ = generate(model, "князь андрей", 60, temperature=0.8)
            samples.append({"step": step, "text": text})

        print(f"Шаг {step}/{steps} | train: {train_loss:.3f} | val: {val_loss:.3f}",
              flush=True)
        model.train()

    started = perf_counter()
    record(0, measure_loss(model, x_start, y_start))
    loss_sum = 0.0

    for step in range(1, steps + 1):
        x, y = sample_windows(train_ids, batch_size, context_size, train_rng)

        optimizer.zero_grad()
        loss = next_token_loss(model, x.to(device), y.to(device))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if step % interval == 0:
            record(step, loss_sum / interval)
            loss_sum = 0.0

    model.eval()
    return model, curve, samples, (perf_counter() - started) / 60


@torch.no_grad()
def compare_models(transformer, train_words, eval_words):
    bigram = BigramModel()
    bigram.fit(train_words)
    bigram_loss = -bigram.logprob(eval_words, alpha=0.01)

    device = next(transformer.parameters()).device
    ids = torch.tensor(encode_words(eval_words), device=device)
    first_target = len(encode_words(eval_words[:1]))
    context_size = transformer.context_size
    stride = context_size // 2

    transformer_loss = 0.0

    for start in range(first_target, len(ids), stride):
        stop = min(start + stride, len(ids))
        left = max(0, stop - 1 - context_size)

        inputs = ids[left:stop - 1].unsqueeze(0)
        targets = ids[start:stop]
        logits = transformer(inputs)[0, -len(targets):]

        transformer_loss += F.cross_entropy(
            logits, targets, reduction="sum"
        ).item()

    result = {"words": len(eval_words)}

    for name, tokens, loss in (
        ("bigram", len(eval_words) - 1, bigram_loss),
        ("tiny", len(ids) - first_target, transformer_loss),
    ):
        average = loss / len(eval_words)
        result[name] = {
            "tokens": tokens,
            "nats_total": loss,
            "nats_per_word": average,
            "ppl_per_word": math.exp(average),
        }

    result["bigram"]["alpha"] = 0.01
    return result


def first_repeat(ids):
    seen = set()

    for start in range(len(ids) - 3):
        group = tuple(ids[start:start + 4])
        if group in seen:
            return start
        seen.add(group)

    return None


def build_generation_report(model, max_new_tokens):
    prompt = "князь андрей"
    modes = [
        ("greedy", 0.0, None),
        ("t0.7", 0.7, None),
        ("t1.5", 1.5, None),
        ("top_p0.9", 1.0, 0.9),
    ]
    runs = []

    for name, temperature, top_p in modes:
        text, ids = generate(model, prompt, max_new_tokens, temperature, top_p)
        runs.append({
            "name": name,
            "temperature": temperature,
            "top_p": top_p,
            "text": text,
            "ids": ids,
            "n_tokens": len(ids),
            "loop_start": first_repeat(ids) if name == "greedy" else None,
        })

    return {
        "model": "tiny",
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "runs": runs,
    }

def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    global ranks, vocabulary, token_to_id, word_id_cache

    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--bpe", type=Path, default=ROOT / "pt01" / "artifacts" / "bpe.json"
    )
    args = parser.parse_args()

    if not args.bpe.is_file():
        parser.error(f"Не найден файл BPE: {args.bpe}")

    bpe_data = json.loads(args.bpe.read_text(encoding="utf-8"))
    ranks = {tuple(pair): rank for rank, pair in enumerate(bpe_data["merges"])}
    corpus_words = load_words()

    symbols = set("".join(corpus_words))
    symbols.add("</w>")
    symbols.update(left + right for left, right in ranks)
    vocabulary = sorted(symbols)
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    word_id_cache = {}

    split_index = len(corpus_words) * 9 // 10
    train_words = corpus_words[:split_index]
    val_words = corpus_words[split_index:]
    train_ids = torch.tensor(encode_words(train_words), dtype=torch.long)
    val_ids = torch.tensor(encode_words(val_words), dtype=torch.long)

    model, curve, samples, minutes = train_model(
        train_ids, val_ids, len(vocabulary), fast=args.fast
    )

    config = {
        "vocab_size": len(vocabulary),
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "context": model.context_size,
        "n_params": sum(p.numel() for p in model.parameters()),
        "steps": curve[-1]["step"],
        "batch": 16 if args.fast else 32,
        "lr": 0.001,
        "seed": 42,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "bpe_merges": len(bpe_data["merges"]),
        "train_minutes": minutes,
        "device": next(model.parameters()).device.type,
    }

    generation = build_generation_report(model, 40 if args.fast else 80)
    eval_words = val_words[:300] if args.fast else val_words[:2000]
    evaluation = compare_models(model, train_words, eval_words)

    artifacts = HERE / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    write_json(artifacts / "tiny_config.json", config)
    write_json(artifacts / "samples.json", {
        "prompt": "князь андрей", "temperature": 0.8, "samples": samples,
    })
    write_json(artifacts / "generation.json", generation)
    write_json(artifacts / "eval.json", evaluation)

    with (artifacts / "train_curve.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(curve)

    print(f"Обучение заняло {minutes:.1f} мин.")
    print("Сохранены пять файлов в", artifacts)


if __name__ == "__main__":
    main()
