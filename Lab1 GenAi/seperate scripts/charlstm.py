"""
Generator de text caracter-cu-caracter cu LSTM, antrenat pe poezie romana.

Modul cu toate componentele: date, model, antrenare, generare.
Rulare: python charlstm.py train    (antreneaza si salveaza checkpoint-uri)
        python charlstm.py generate (genereaza din ultimul checkpoint)
"""

import os
import io
import sys
import json
import time
import unicodedata
from pathlib import Path

# Consola Windows foloseste implicit cp1252, care nu poate afisa Ț/ș/ă.
# Fara asta, orice print cu text romanesc ridica UnicodeEncodeError.
# In Jupyter, stdout e un obiect al kernelului fara .buffer si deja UTF-8,
# deci se sare peste rewrapping.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

BASE_DIR = Path(__file__).parent
CORPUS_PATH = BASE_DIR / "corpus_poezie_romana_pd.txt"
CKPT_DIR = BASE_DIR / "checkpoints"
VOCAB_PATH = CKPT_DIR / "vocab.json"

# --- Hiperparametri -------------------------------------------------------
# Calibrati din masuratorile sondei (probe.py) pe acest CPU:
#   units=256 batch=256 seq=100 -> 0.716 sec/step, 35,730 caractere/sec
#   (cel mai bun throughput dintre cele 5 configuratii masurate)
SEQ_LEN = 100        # cate caractere vede modelul inainte sa prezica urmatorul
STRIDE = 5           # decalajul intre ferestre consecutive; mai mare = mai putine
                     # esantioane per trecere, dar mai putina suprapunere redundanta
BATCH_SIZE = 256     # cate secvente se proceseaza simultan intr-un pas de gradient
EMBED_DIM = 64       # dimensiunea vectorului invatat pentru fiecare caracter
LSTM_UNITS = 256     # dimensiunea hidden state-ului; costul creste ~patratic
EPOCHS = 20
STEPS_PER_EPOCH = 250   # "epoca" e aici o unitate de checkpoint, nu o trecere completa
                        # 250 pasi x 0.716 s = ~3 min intre checkpoint-uri
LEARNING_RATE = 2e-3


def load_and_normalize(path: Path = CORPUS_PATH) -> str:
    """Citeste corpusul si normalizeaza Unicode la NFC.

    Fara NFC, 'ș' poate aparea si precompus (U+0219) si ca 's' + cedila
    combinata, ajungand doua intrari distincte in vocabular.
    """
    return unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    return {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)}


def encode(text: str, char_to_idx: dict[str, int]) -> np.ndarray:
    return np.array([char_to_idx[c] for c in text], dtype=np.int32)


def decode(indices, idx_to_char: dict[int, str]) -> str:
    return "".join(idx_to_char[int(i)] for i in indices)


def save_vocab(char_to_idx: dict[str, int]) -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    VOCAB_PATH.write_text(json.dumps(char_to_idx, ensure_ascii=False), encoding="utf-8")


def load_vocab() -> tuple[dict[str, int], dict[int, str]]:
    char_to_idx = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    return char_to_idx, {i: c for c, i in char_to_idx.items()}


def make_dataset(encoded: np.ndarray, seq_len: int = SEQ_LEN,
                 stride: int = STRIDE, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Ferestre glisante peste corpus.

    Fiecare esantion: input = caractere [i, i+seq_len),
    target = aceleasi pozitii decalate cu 1. Modelul prezice, la fiecare
    pas de timp, caracterul urmator — deci un esantion da seq_len predictii.
    """
    starts = np.arange(0, len(encoded) - seq_len - 1, stride, dtype=np.int64)
    windows = encoded[starts[:, None] + np.arange(seq_len + 1, dtype=np.int64)[None, :]]

    ds = tf.data.Dataset.from_tensor_slices((windows[:, :-1], windows[:, 1:]))
    ds = ds.shuffle(10_000).batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE).repeat()


def build_model(vocab_size: int, embed_dim: int = EMBED_DIM,
                units: int = LSTM_UNITS) -> keras.Model:
    """Embedding -> LSTM -> Dense.

    Dense produce logits (scoruri nenormalizate) peste tot vocabularul,
    la fiecare pas de timp. Softmax-ul se aplica in loss si la generare.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(None,), dtype="int32"),
        keras.layers.Embedding(vocab_size, embed_dim),
        keras.layers.LSTM(units, return_sequences=True),
        keras.layers.Dense(vocab_size),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    return model


def pick_next(probs: np.ndarray, strategy: str = "sample",
              min_prob: float = 0.5, top_k: int = 5) -> int:
    """Alege indexul caracterului urmator dintr-o distributie de probabilitati.

    'sample'   — trage la sorti proportional cu probabilitatea (coada inclusa)
    'argmax'   — mereu cel mai probabil; determinist, intra in bucle
    'confident'— daca un caracter depaseste min_prob, il ia direct;
                 altfel esantioneaza doar dintre primii top_k
    """
    if strategy == "argmax":
        return int(np.argmax(probs))

    if strategy == "confident":
        best = int(np.argmax(probs))
        if probs[best] >= min_prob:
            return best
        top_idx = np.argsort(probs)[::-1][:top_k]
        renormalized = probs[top_idx] / probs[top_idx].sum()
        return int(np.random.choice(top_idx, p=renormalized))

    return int(np.random.choice(len(probs), p=probs))


def generate(model: keras.Model, seed: str, char_to_idx: dict[str, int],
             idx_to_char: dict[int, str], n_chars: int = 400,
             temperature: float = 0.8, strategy: str = "sample") -> str:
    """Generare autoregresiva.

    Bucla: modelul primeste secventa de pana acum, produce logits pentru
    caracterul urmator, se imparte la temperature, se esantioneaza din
    distributie, iar caracterul ales se adauga la secventa si intra in
    iteratia urmatoare.

    temperature < 1 ascute distributia (mai conservator, se repeta);
    temperature > 1 o aplatizeaza (mai divers, mai multe non-cuvinte).
    """
    seed = unicodedata.normalize("NFC", seed)
    context = [char_to_idx[c] for c in seed if c in char_to_idx]
    if not context:
        raise ValueError("seed-ul nu contine niciun caracter din vocabular")

    generated = []
    for _ in range(n_chars):
        window = context[-SEQ_LEN:]
        logits = model(tf.constant([window], dtype=tf.int32), training=False)
        probs = tf.nn.softmax(logits[0, -1] / temperature).numpy().astype(np.float64)
        probs /= probs.sum()  # softmax in float32 nu insumeaza exact 1.0

        next_idx = pick_next(probs, strategy)
        context.append(next_idx)
        generated.append(next_idx)

    return seed + decode(generated, idx_to_char)


def train() -> None:
    CKPT_DIR.mkdir(exist_ok=True)

    text = load_and_normalize()
    char_to_idx, idx_to_char = build_vocab(text)
    save_vocab(char_to_idx)
    encoded = encode(text, char_to_idx)

    print(f"Corpus: {len(text):,} caractere | vocabular: {len(char_to_idx)}")

    n_windows = (len(encoded) - SEQ_LEN - 1) // STRIDE
    samples_per_epoch = STEPS_PER_EPOCH * BATCH_SIZE
    print(f"Ferestre totale: {n_windows:,} | esantioane/epoca: {samples_per_epoch:,} "
          f"({samples_per_epoch / n_windows:.1%} din date)")
    print(f"Treceri reale prin corpus dupa {EPOCHS} epoci: "
          f"{EPOCHS * samples_per_epoch / n_windows:.2f}\n")

    ds = make_dataset(encoded)
    model = build_model(len(char_to_idx))
    model.summary()

    # Verificare a caii de generare INAINTE de antrenare: output-ul va fi
    # gunoi, dar daca bucla crapa, se vede acum si nu la finalul antrenarii.
    print("\nTest generare pe model neantrenat (output asteptat: gunoi):")
    print(repr(generate(model, "Adio", char_to_idx, idx_to_char, n_chars=60)))

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(CKPT_DIR / "epoch_{epoch:02d}.keras"),
            save_freq="epoch",
        ),
        keras.callbacks.CSVLogger(str(CKPT_DIR / "history.csv")),
    ]

    print(f"\nAntrenare: {EPOCHS} epoci x {STEPS_PER_EPOCH} pasi\n")
    t0 = time.perf_counter()
    model.fit(ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, callbacks=callbacks)
    print(f"\nDurata totala: {(time.perf_counter() - t0) / 60:.1f} min")

    model.save(CKPT_DIR / "final.keras")


def generate_from_checkpoint(ckpt: str = "final.keras") -> None:
    char_to_idx, idx_to_char = load_vocab()
    model = keras.models.load_model(CKPT_DIR / ckpt)

    for temp in (0.2, 0.5, 0.8, 1.2):
        print(f"\n{'=' * 60}\ntemperature = {temp}\n{'=' * 60}")
        print(generate(model, "Adio\n\n", char_to_idx, idx_to_char,
                       n_chars=300, temperature=temp))


def interactive(ckpt: str = "final.keras") -> None:
    """Mod interactiv: dai un inceput de text, modelul il continua.

    Nu e conversatie — modelul nu raspunde la intrebari, doar continua
    secventa de caractere pe care i-o dai.
    """
    char_to_idx, idx_to_char = load_vocab()
    model = keras.models.load_model(CKPT_DIR / ckpt)

    temperature = 0.8
    n_chars = 300
    strategy = "sample"

    print(f"Model: {ckpt} | temperature={temperature} | {n_chars} caractere")
    print("Comenzi:  /temp 1.2   /len 500   /mod confident   /quit")
    print("Orice altceva devine inceputul textului pe care il continua.\n")

    while True:
        try:
            seed = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not seed:
            continue
        if seed == "/quit":
            return
        if seed.startswith("/temp "):
            temperature = float(seed.split()[1])
            print(f"temperature = {temperature}\n")
            continue
        if seed.startswith("/len "):
            n_chars = int(seed.split()[1])
            print(f"lungime = {n_chars}\n")
            continue
        if seed.startswith("/mod "):
            choice = seed.split()[1]
            if choice not in ("confident", "sample", "argmax"):
                print("moduri: confident | sample | argmax\n")
                continue
            strategy = choice
            print(f"strategie = {strategy}\n")
            continue

        unknown = {c for c in unicodedata.normalize("NFC", seed) if c not in char_to_idx}
        if unknown:
            print(f"(caractere ignorate, nu sunt in vocabular: {sorted(unknown)})")

        print()
        print(generate(model, seed, char_to_idx, idx_to_char,
                       n_chars=n_chars, temperature=temperature, strategy=strategy))
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train()
    elif cmd == "generate":
        generate_from_checkpoint(*sys.argv[2:3])
    elif cmd == "chat":
        interactive(*sys.argv[2:3])
    else:
        print(f"comanda necunoscuta: {cmd} (foloseste 'train', 'generate' sau 'chat')")
