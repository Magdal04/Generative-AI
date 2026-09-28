"""
Vizualizeaza decizia modelului la fiecare caracter generat.

La fiecare pas arata: contextul, primii 5 candidati cu probabilitatile lor,
si care a fost ales prin esantionare. Face vizibil ce inseamna "modelul
prezice o distributie, nu un caracter".

Rulare: python inspect_generation.py            (pas cu pas, 40 caractere)
        python inspect_generation.py --temp 0.3 (alta temperatura)
        python inspect_generation.py --slow     (pauza intre pasi)
"""

import argparse
import time
import unicodedata

import numpy as np
import tensorflow as tf
from tensorflow import keras

from charlstm import CKPT_DIR, SEQ_LEN, load_vocab

BAR_WIDTH = 30


def show_char(c: str) -> str:
    """Face vizibile caracterele invizibile."""
    return {"\n": "\\n", " ": "␣", "\t": "\\t"}.get(c, c)


def step_by_step(model, seed: str, char_to_idx, idx_to_char,
                 n_chars: int = 40, temperature: float = 0.8,
                 top_k: int = 5, delay: float = 0.0) -> str:
    seed = unicodedata.normalize("NFC", seed)
    context = [char_to_idx[c] for c in seed if c in char_to_idx]
    generated = ""

    for step in range(n_chars):
        window = context[-SEQ_LEN:]
        logits = model(tf.constant([window], dtype=tf.int32), training=False)
        raw_logits = logits[0, -1].numpy()

        scaled = raw_logits / temperature
        probs = tf.nn.softmax(scaled).numpy()

        chosen = int(tf.random.categorical(scaled[None, :], num_samples=1)[0, 0])
        context.append(chosen)
        generated += idx_to_char[chosen]

        top_idx = np.argsort(probs)[::-1][:top_k]

        tail = "".join(idx_to_char[i] for i in context[-40:-1])
        print(f"\n{'─' * 64}")
        print(f"pas {step + 1}/{n_chars}   context: ...{tail.replace(chr(10), '⏎')}")
        print(f"{'─' * 64}")

        for rank, idx in enumerate(top_idx, 1):
            p = probs[idx]
            bar = "█" * int(p * BAR_WIDTH)
            mark = " ◄ ALES" if idx == chosen else ""
            print(f"  {rank}. '{show_char(idx_to_char[idx]):>3}'  {p:6.1%} {bar:<{BAR_WIDTH}}{mark}")

        if chosen not in top_idx:
            p = probs[chosen]
            print(f"     '{show_char(idx_to_char[chosen]):>3}'  {p:6.1%} "
                  f"{'':<{BAR_WIDTH}} ◄ ALES (in afara top {top_k})")

        # Entropia arata cat de "sigur" e modelul: mica = distributie ascutita,
        # mare = modelul ezita intre multe optiuni.
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        print(f"\n  entropie: {entropy:.2f} biti   "
              f"(0 = sigur pe un caracter, {np.log2(len(probs)):.1f} = total nesigur)")

        if delay:
            time.sleep(delay)

    return seed + generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="ARVINTE: ")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--ckpt", default="final.keras")
    parser.add_argument("--slow", action="store_true")
    args = parser.parse_args()

    char_to_idx, idx_to_char = load_vocab()
    model = keras.models.load_model(CKPT_DIR / args.ckpt)

    print(f"model: {args.ckpt} | temperature: {args.temp} | seed: {args.seed!r}")

    result = step_by_step(
        model, args.seed, char_to_idx, idx_to_char,
        n_chars=args.n, temperature=args.temp,
        delay=0.4 if args.slow else 0.0,
    )

    print(f"\n{'=' * 64}\nTEXT FINAL\n{'=' * 64}\n{result}")


if __name__ == "__main__":
    main()
