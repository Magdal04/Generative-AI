"""
Sonda de masurare: cat dureaza un pas de antrenare pe ACEST CPU.

Nu antreneaza nimic util. Ruleaza 50 de pasi cu mai multe configuratii,
masoara sec/step real, si scrie un tabel. Din numerele astea se calibreaza
toti hiperparametrii rularii lungi.

Ruleaza: python probe.py
"""

import os
import time
import unicodedata
from pathlib import Path

# Numarul de thread-uri trebuie fixat INAINTE de importul TensorFlow,
# altfel TF autotuneaza si masuratorile devin necomparabile intre ele.
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

CORPUS_PATH = Path(__file__).parent / "corpus_poezie_romana_pd.txt"
PROBE_STEPS = 50
SUBSET_CHARS = 300_000


def load_and_normalize(path: Path) -> str:
    """Citeste corpusul si normalizeaza Unicode la forma NFC.

    Fara NFC, 'ș' poate exista si precompus (U+0219) si ca 's' + cedila
    combinata (U+0073 U+0327). Nenormalizat ajung doua intrari diferite
    in vocabular pentru acelasi caracter vizual.
    """
    raw = path.read_text(encoding="utf-8")
    return unicodedata.normalize("NFC", raw)


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


def make_dataset(
    encoded: np.ndarray, seq_len: int, stride: int, batch_size: int
) -> tf.data.Dataset:
    """Ferestre glisante: input = caractere [i : i+seq_len],
    target = aceleasi caractere decalate cu 1 pozitie."""
    starts = np.arange(0, len(encoded) - seq_len - 1, stride, dtype=np.int64)
    window_idx = starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]
    windows = encoded[window_idx]

    inputs = windows[:, :-1]
    targets = windows[:, 1:]

    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    ds = ds.shuffle(10_000).batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE).repeat()


def build_model(vocab_size: int, embed_dim: int, units: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Input(shape=(None,), dtype="int32"),
            keras.layers.Embedding(vocab_size, embed_dim),
            keras.layers.LSTM(units, return_sequences=True),
            keras.layers.Dense(vocab_size),
        ]
    )


def time_config(
    encoded: np.ndarray,
    vocab_size: int,
    units: int,
    batch_size: int,
    seq_len: int,
    embed_dim: int = 64,
    stride: int = 5,
) -> dict:
    ds = make_dataset(encoded, seq_len, stride, batch_size)
    model = build_model(vocab_size, embed_dim, units)
    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    it = iter(ds)
    # Primii pasi includ compilarea grafului si alocarea — nu se cronometreaza.
    for _ in range(3):
        model.train_on_batch(*next(it))

    t0 = time.perf_counter()
    for _ in range(PROBE_STEPS):
        model.train_on_batch(*next(it))
    elapsed = time.perf_counter() - t0

    sec_per_step = elapsed / PROBE_STEPS
    chars_per_sec = (batch_size * seq_len) / sec_per_step

    return {
        "units": units,
        "batch": batch_size,
        "seq_len": seq_len,
        "params": model.count_params(),
        "sec_per_step": sec_per_step,
        "chars_per_sec": chars_per_sec,
    }


def main() -> None:
    print(f"TF {tf.__version__} | intra-op threads: {os.environ['TF_NUM_INTRAOP_THREADS']}")

    text = load_and_normalize(CORPUS_PATH)
    char_to_idx, _ = build_vocab(text)
    vocab_size = len(char_to_idx)
    print(f"Corpus: {len(text):,} caractere | vocabular: {vocab_size} caractere unice\n")

    subset = text[:SUBSET_CHARS]
    encoded = np.array([char_to_idx[c] for c in subset], dtype=np.int32)

    configs = [
        dict(units=128, batch_size=128, seq_len=100),
        dict(units=256, batch_size=128, seq_len=100),
        dict(units=256, batch_size=256, seq_len=100),
        dict(units=256, batch_size=128, seq_len=50),
        dict(units=512, batch_size=128, seq_len=100),
    ]

    print(f"{'units':>6} {'batch':>6} {'seq':>5} {'params':>10} {'sec/step':>10} {'chars/sec':>12}")
    print("-" * 56)

    results = []
    for cfg in configs:
        r = time_config(encoded, vocab_size, **cfg)
        results.append(r)
        print(
            f"{r['units']:>6} {r['batch']:>6} {r['seq_len']:>5} {r['params']:>10,} "
            f"{r['sec_per_step']:>10.3f} {r['chars_per_sec']:>12,.0f}"
        )

    print("\nExtrapolare pentru corpusul complet (1.49M caractere, stride 5):")
    n_windows = (len(text) - 101) // 5
    for r in results:
        steps_per_pass = n_windows // r["batch"]
        minutes = steps_per_pass * r["sec_per_step"] / 60
        print(
            f"  units={r['units']:>3} batch={r['batch']:>3} seq={r['seq_len']:>3}: "
            f"{steps_per_pass:>5,} pasi/trecere = {minutes:>6.1f} min/trecere completa"
        )


if __name__ == "__main__":
    main()
