"""
Evaluari pentru raport si examenul oral.

1. Memorare: verifica daca textul generat contine n-grame copiate verbatim
   din corpus. Raspunde la intrebarea "modelul memoreaza sau generalizeaza?".
2. Distributia diacriticelor: compara frecventa ă/â/î/ș/ț in text generat
   fata de corpus. Arata daca modelul a invatat ortografia romana.
3. Curba de invatare: text generat din checkpoint-uri succesive, ca sa se
   vada progresia de la gunoi la structura.

Rulare: python evaluate.py
"""

import unicodedata
from collections import Counter
from pathlib import Path

# Importul lui charlstm seteaza deja stdout pe UTF-8 — nu se re-wrappeaza aici,
# fiindca al doilea wrapper l-ar inchide pe primul.

from charlstm import (
    CKPT_DIR,
    generate,
    load_and_normalize,
    load_vocab,
)

DIACRITICE = "ăâîșț"


def longest_common_ngram(generated: str, corpus: str, max_n: int = 40) -> int:
    """Cea mai lunga secventa din `generated` care apare verbatim in corpus.

    Cauta binar pe lungime: daca exista o potrivire de lungime n, incearca
    mai lung; altfel mai scurt.
    """
    def exists(n: int) -> bool:
        return any(
            generated[i:i + n] in corpus
            for i in range(len(generated) - n + 1)
        )

    lo, hi = 0, min(max_n, len(generated))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if exists(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def char_distribution(text: str, chars: str = DIACRITICE) -> dict[str, float]:
    """Frecventa fiecarui caracter, ca procent din total."""
    counts = Counter(text.lower())
    total = len(text)
    return {c: 100 * counts[c] / total for c in chars}


def report_memorization(generated: str, corpus: str) -> None:
    n = longest_common_ngram(generated, corpus)
    print(f"Cea mai lunga secventa copiata verbatim din corpus: {n} caractere")
    if n >= 30:
        print("  -> lung. Modelul reproduce fragmente memorate.")
    elif n >= 12:
        print("  -> moderat. Normal pentru expresii frecvente in poezie.")
    else:
        print("  -> scurt. Modelul compune, nu copiaza.")


def report_diacritics(generated: str, corpus: str) -> None:
    gen_dist = char_distribution(generated)
    corp_dist = char_distribution(corpus)

    print(f"\n{'caracter':>10} {'generat %':>12} {'corpus %':>12} {'raport':>10}")
    print("-" * 48)
    for c in DIACRITICE:
        g, k = gen_dist[c], corp_dist[c]
        ratio = g / k if k > 0 else float("nan")
        print(f"{c:>10} {g:>12.3f} {k:>12.3f} {ratio:>10.2f}")
    print("\nRaport apropiat de 1.00 = modelul a invatat frecventa corecta.")


def report_learning_curve(char_to_idx, idx_to_char, corpus: str) -> None:
    """Genereaza din checkpoint-uri succesive ca sa se vada progresia."""
    from tensorflow import keras

    ckpts = sorted(CKPT_DIR.glob("epoch_*.keras"))
    if not ckpts:
        print("Niciun checkpoint gasit.")
        return

    picks = [ckpts[0], ckpts[len(ckpts) // 2], ckpts[-1]]
    for ckpt in picks:
        model = keras.models.load_model(ckpt)
        text = generate(model, "Adio\n\n", char_to_idx, idx_to_char,
                        n_chars=200, temperature=0.8)
        print(f"\n{'=' * 60}\n{ckpt.name}\n{'=' * 60}\n{text}")


def main() -> None:
    from tensorflow import keras

    corpus = load_and_normalize()
    char_to_idx, idx_to_char = load_vocab()
    model = keras.models.load_model(CKPT_DIR / "final.keras")

    print("Generez 1000 de caractere la temperature=0.8 pentru evaluare...\n")
    generated = generate(model, "Adio\n\n", char_to_idx, idx_to_char,
                         n_chars=1000, temperature=0.8)

    print(f"{'=' * 60}\nTEXT GENERAT\n{'=' * 60}\n{generated}\n")

    print(f"{'=' * 60}\nMEMORARE\n{'=' * 60}")
    report_memorization(generated, corpus)

    print(f"\n{'=' * 60}\nDISTRIBUTIA DIACRITICELOR\n{'=' * 60}")
    report_diacritics(generated, corpus)

    print(f"\n{'=' * 60}\nPROGRESIA INVATARII\n{'=' * 60}")
    report_learning_curve(char_to_idx, idx_to_char, corpus)


if __name__ == "__main__":
    main()
