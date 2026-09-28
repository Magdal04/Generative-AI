"""Antreneaza judecatorul pe MNIST si il salveaza in modele/judecator.keras."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

from src.judecator import CALE_JUDECATOR, construieste_judecator, evalueaza

def main():
    tf.random.set_seed(0)
    (xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
    prep = lambda x: ((x.astype(np.float32) - 127.5) / 127.5).reshape(-1, 784)
    xtr, xte = prep(xtr), prep(xte)

    J = construieste_judecator()
    J.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    J.fit(xtr, ytr, epochs=3, batch_size=128, validation_split=0.1, verbose=2)
    _, acc = J.evaluate(xte, yte, verbose=0)
    print(f"acuratete pe test (10 000 cifre nevazute): {acc:.4f}")
    J.save(CALE_JUDECATOR)

    # Liniile de baza: ce scor da judecatorul pe cifre reale si pe zgomot
    r = evalueaza(J, xte)
    print(f"cifre reale  -> claritate {r['claritate']:.3f}, distributie {np.round(r['distributie'], 3)}")
    zg = np.random.default_rng(0).uniform(-1, 1, (10000, 784)).astype(np.float32)
    r = evalueaza(J, zg)
    print(f"zgomot pur   -> claritate {r['claritate']:.3f}, distributie {np.round(r['distributie'], 3)}")
    medie = np.repeat(xtr.mean(0, keepdims=True), 10, 0)
    r = evalueaza(J, medie)
    print(f"cifra medie  -> claritate {r['claritate']:.3f}, prezisa ca {r['cifre'][0]}")


if __name__ == "__main__":
    main()
