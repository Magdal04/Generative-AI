"""
Interfata de chat cu generare vizibila caracter cu caracter.

Scrii un inceput de text, modelul il continua incet, o litera pe secunda,
si sub el se construieste array-ul de probabilitati: pentru fiecare
caracter scris, cat de sigur a fost modelul cand l-a ales.

Rulare: python chat_live.py
        python chat_live.py --delay 0.5 --temp 1.2
"""

import argparse
import sys
import time
import unicodedata

import numpy as np
import tensorflow as tf
from tensorflow import keras

from charlstm import CKPT_DIR, SEQ_LEN, load_vocab, pick_next

MAX_CHARS = 120


def stream_generate(model, seed: str, char_to_idx, idx_to_char,
                    n_chars: int, temperature: float, delay: float,
                    strategy: str = "confident") -> None:
    """Genereaza caracter cu caracter, afisand pe masura ce scrie."""
    seed = unicodedata.normalize("NFC", seed)
    context = [char_to_idx[c] for c in seed if c in char_to_idx]

    if not context:
        print("  (niciun caracter din seed nu e in vocabular)\n")
        return

    probabilities: list[int] = []
    n_confident = 0

    print(f"\n  \033[2m{seed}\033[0m", end="", flush=True)

    for _ in range(n_chars):
        window = context[-SEQ_LEN:]
        logits = model(tf.constant([window], dtype=tf.int32), training=False)
        probs = tf.nn.softmax(logits[0, -1] / temperature).numpy().astype(np.float64)
        probs /= probs.sum()

        if probs.max() >= 0.5:
            n_confident += 1

        chosen = pick_next(probs, strategy)
        context.append(chosen)

        probabilities.append(round(float(probs[chosen]) * 100))

        print(idx_to_char[chosen], end="", flush=True)
        time.sleep(delay)

    print(f"\n\n  sanse: {probabilities}")
    avg = sum(probabilities) / len(probabilities)
    print(f"  medie: {avg:.0f}%  |  min: {min(probabilities)}%  "
          f"max: {max(probabilities)}%")
    print(f"  peste 50%: {n_confident}/{n_chars} pasi "
          f"({n_confident / n_chars:.0%}) — restul au ales din top 5\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=1.0,
                        help="secunde intre caractere (implicit 1.0)")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=MAX_CHARS)
    parser.add_argument("--ckpt", default="final.keras")
    parser.add_argument("--strategy", default="confident",
                        choices=["confident", "sample", "argmax"])
    args = parser.parse_args()

    n_chars = min(args.n, MAX_CHARS)
    temperature = args.temp
    delay = args.delay
    strategy = args.strategy

    char_to_idx, idx_to_char = load_vocab()
    model = keras.models.load_model(CKPT_DIR / args.ckpt)

    print(f"\n  \033[1mchar-LSTM\033[0m  {args.ckpt}  |  {len(char_to_idx)} caractere in vocabular")
    print(f"  temperature={temperature}  delay={delay}s  max={n_chars} caractere")
    print(f"  strategie={strategy} (peste 50% ia direct, altfel top 5)")
    print("  \033[2m/temp 1.2   /delay 0.3   /len 40   /mod sample   /quit\033[0m\n")

    while True:
        try:
            seed = input("\033[1m>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not seed:
            continue
        if seed == "/quit":
            return
        if seed.startswith("/temp "):
            temperature = float(seed.split()[1])
            print(f"  temperature = {temperature}\n")
            continue
        if seed.startswith("/delay "):
            delay = float(seed.split()[1])
            print(f"  delay = {delay}s\n")
            continue
        if seed.startswith("/len "):
            n_chars = min(int(seed.split()[1]), MAX_CHARS)
            print(f"  lungime = {n_chars} caractere\n")
            continue
        if seed.startswith("/mod "):
            choice = seed.split()[1]
            if choice not in ("confident", "sample", "argmax"):
                print("  moduri: confident | sample | argmax\n")
                continue
            strategy = choice
            print(f"  strategie = {strategy}\n")
            continue

        stream_generate(model, seed, char_to_idx, idx_to_char,
                        n_chars, temperature, delay, strategy)


if __name__ == "__main__":
    main()
