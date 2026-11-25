import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from urllib.request import urlopen


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CORPUS_URL = "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/master/books_before/WarAndPeace.txt"


def load_words():
    with urlopen(CORPUS_URL, timeout=60) as response:
        text = response.read().decode("utf-8-sig")

    normalized = text.lower().replace("ё", "е")
    return re.findall(r"[а-я]+(?:-[а-я]+)*", normalized)


class BigramModel:
    def fit(self, training_words):
        self.bigram_counts = {}
        self.vocabulary = set(training_words)
        self.vocabulary.add("<unk>")

        for current, following in zip(training_words, training_words[1:]):
            if current not in self.bigram_counts:
                self.bigram_counts[current] = {}

            counts = self.bigram_counts[current]
            counts[following] = counts.get(following, 0) + 1

    def prob(self, current, following, alpha=0.0):
        counts = self.bigram_counts.get(current, {})
        count = counts.get(following, 0)
        denominator = sum(counts.values()) + alpha * len(self.vocabulary)

        if denominator == 0:
            return 0.0

        return (count + alpha) / denominator

    def logprob(self, sentence_words, alpha=0.0):
        total = 0.0

        for current, following in zip(sentence_words, sentence_words[1:]):
            probability = self.prob(current, following, alpha)

            if probability == 0:
                return float("-inf")

            total += math.log(probability)

        return total

    def generate(self, start, length, temperature=1.0, seed=42):
        rng = random.Random(seed)
        result = [start]

        for _ in range(length - 1):
            counts = self.bigram_counts.get(result[-1], {})

            if not counts:
                break

            weights = [count ** (1 / temperature) for count in counts.values()]
            next_word = rng.choices(list(counts), weights=weights, k=1)[0]
            result.append(next_word)

        return result


def merge_pair(symbols, pair):
    result = []
    i = 0

    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1

    return result


def count_symbol_pairs(forms, frequencies):
    counts = Counter()

    for word, symbols in forms.items():
        for left, right in zip(symbols, symbols[1:]):
            counts[(left, right)] += frequencies[word]

    return counts


def train_bpe(frequencies, num_merges, min_frequency=5):
    forms = {}

    for word, frequency in frequencies.items():
        if frequency >= min_frequency:
            forms[word] = list(word) + ["</w>"]

    merges = []

    for _ in range(num_merges):
        counts = count_symbol_pairs(forms, frequencies)
        if not counts:
            break

        best_pair, _ = counts.most_common(1)[0]
        merges.append(best_pair)

        for word, symbols in forms.items():
            forms[word] = merge_pair(symbols, best_pair)

    return merges


def encode_word(word, ranks):
    symbols = list(word) + ["</w>"]

    while True:
        candidates = []

        for i, pair in enumerate(zip(symbols, symbols[1:])):
            if pair in ranks:
                candidates.append((ranks[pair], i))

        if not candidates:
            return symbols

        _, i = min(candidates)
        symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]


def build_bigram_report(corpus_words, train_words, val_words, word_counts):
    model = BigramModel()
    model.fit(train_words)

    original = val_words[:12]
    shuffled = original.copy()
    random.Random(42).shuffle(shuffled)

    unseen_count = 0
    for current, following in zip(original, original[1:]):
        if model.prob(current, following) == 0:
            unseen_count += 1

    perplexities = {}
    for alpha in (1.0, 0.1, 0.01):
        score = model.logprob(val_words, alpha=alpha)
        perplexities[f"{alpha:g}"] = math.exp(-score / (len(val_words) - 1))

    generation = []
    for temperature in (0.5, 1.0, 2.0):
        for start in ("князь", "она"):
            sample = model.generate(start, 30, temperature=temperature)
            generation.append({
                "temperature": temperature,
                "start": start,
                "text": " ".join(sample),
            })

    return {
        "corpus": "Война и мир",
        "words": len(corpus_words),
        "train_words": len(train_words),
        "val_words": len(val_words),
        "vocab_size": len(word_counts),
        "top20": word_counts.most_common(20),
        "perplexity": perplexities,
        "likelihood": {
            "sentence": " ".join(original),
            "shuffled": " ".join(shuffled),
            "alpha": 0.01,
            "logprob_sentence": model.logprob(original, alpha=0.01),
            "logprob_shuffled": model.logprob(shuffled, alpha=0.01),
            "unseen_pairs_in_sentence": unseen_count,
        },
        "generation": generation,
    }


def build_bpe_report(word_counts, num_merges):
    merges = train_bpe(word_counts, num_merges)
    ranks = {pair: rank for rank, pair in enumerate(merges)}

    vocabulary = set("".join(word_counts))
    vocabulary.add("</w>")
    vocabulary.update(left + right for left, right in merges)

    total_words = sum(word_counts.values())
    total_chars = 0
    total_tokens = 0

    for word, frequency in word_counts.items():
        total_chars += len(word) * frequency
        total_tokens += len(encode_word(word, ranks)) * frequency

    return {
        "num_merges": len(merges),
        "vocab_size": len(vocabulary),
        "merges": merges,
        "stats": {
            "chars": total_chars,
            "words": total_words,
            "tokens_bpe": total_tokens,
            "tokens_per_word": total_tokens / total_words,
            "compression_vs_chars": total_chars / total_tokens,
        },
        "examples": {
            word: encode_word(word, ranks)
            for word in ("наполеон", "сказала", "неудовольствие")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    corpus_words = load_words()
    split_index = len(corpus_words) * 9 // 10
    train_words = corpus_words[:split_index]
    val_words = corpus_words[split_index:]
    word_counts = Counter(corpus_words)

    bigram_report = build_bigram_report(corpus_words, train_words, val_words, word_counts)
    bpe_report = build_bpe_report(word_counts, 200 if args.fast else 1500)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for filename, report in (("bigram.json", bigram_report), ("bpe.json", bpe_report)):
        path = ARTIFACTS / filename
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Создан файл:", path)

    print("Слияний BPE:", bpe_report["num_merges"])


if __name__ == "__main__":
    main()
