from pathlib import Path

CORPUS_PATH = Path(__file__).parent / "corpus_poezie_romana_pd.txt"


def load_corpus(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """
    Contract:
      - extrage mulțimea caracterelor unice din `text`
      - returnează (char_to_idx, idx_to_char), două mapări inverse una alteia
      - ordinea indicilor trebuie să fie deterministă (rulezi de două ori,
        primești aceleași mapări)
    """
    # TODO(eu): implementează. ~5-8 linii.
    sorted_alphabet = sorted(set(text))
    char_to_idx = {sorted_alphabet : i for i, sorted_alphabet in enumerate(sorted_alphabet)}
    idx_to_char = {i : sorted_alphabet for i, sorted_alphabet in enumerate(sorted_alphabet)}
    return (char_to_idx, idx_to_char)



def encode(text: str, char_to_idx: dict[str, int]) -> list[int]:
    """
    Contract:
      - transformă fiecare caracter din `text` în indexul lui din char_to_idx
      - ridică KeyError dacă apare un caracter care nu e în vocabular
    """
    # TODO(eu): implementează. ~2-4 linii.
    transformed = [char_to_idx[c] for c in text]
    print(transformed)


def decode(indices: list[int], idx_to_char: dict[int, str]) -> str:
    """
    Contract:
      - inversul lui `encode`: din listă de indici, reconstruiește string-ul
    """
    # TODO(eu): implementează. ~2-4 linii.
    raise NotImplementedError


if __name__ == "__main__":
    corpus = load_corpus(CORPUS_PATH)
    char_to_idx, idx_to_char = build_vocab(corpus)
    print(f"Corpus: {len(corpus)} caractere, vocabular: {len(char_to_idx)} caractere unice")

    sample = corpus[:20]
    encoded = encode(sample, char_to_idx)
    #decoded = decode(encoded, idx_to_char)
    #assert decoded == sample, f"round-trip eșuat: {decoded!r} != {sample!r}"
    #print("round-trip OK:", encoded)
