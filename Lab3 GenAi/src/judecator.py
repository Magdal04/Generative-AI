"""
Judecatorul extern: un clasificator MNIST antrenat separat, care nu participa
la jocul GAN. Il folosim ca sa masuram doua lucruri pe cifrele generate:
  claritate  - cat de sigur e judecatorul pe fiecare imagine (max softmax)
  varietate  - cat de des apare fiecare cifra in tot setul generat

Primeste imagini in [-1, 1], aceeasi scara ca iesirea lui G, ca sa nu avem
nevoie de conversii intre GAN si judecator.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

CALE_JUDECATOR = "modele/judecator.keras"


def construieste_judecator():
    return tf.keras.Sequential(
        [
            layers.Input((784,)),
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="judecator",
    )


def incarca_judecator():
    return tf.keras.models.load_model(CALE_JUDECATOR)


def scor_inception(p, n_split=10, eps=1e-16):
    """
    Inception Score din curs: exp(E_x[KL(p(y|x) || p(y))]), cu judecatorul
    MNIST in locul lui InceptionV3. p: (n, 10) probabilitati softmax.
    p(y) se calculeaza separat in fiecare parte (exemplul 3_3 din curs il
    calculeaza o singura data, pe tot setul).
    Intre 1 (o singura cifra sau nesiguranta totala) si 10 (toate cifrele, sigur).
    """
    scoruri = []
    for parte in np.array_split(p, n_split):
        py = parte.mean(0)
        kl = (parte * (np.log(parte + eps) - np.log(py + eps))).sum(1)
        scoruri.append(np.exp(kl.mean()))
    return float(np.mean(scoruri)), float(np.std(scoruri))


def descompune_is(p, eps=1e-16):
    """
    IS pe tot setul, fara parti, desfacut exact in cele doua jumatati:
      IS = cifre_efective / ezitare
      cifre_efective = exp(H(p(y)))        cate cifre produce G efectiv (10 = uniform)
      ezitare        = exp(E_x[H(p(y|x))]) intre cate cifre ezita judecatorul pe o imagine (1 = sigur)
    """
    py = p.mean(0)
    cifre_efective = np.exp(-(py * np.log(py + eps)).sum())
    ezitare = np.exp(-(p * np.log(p + eps)).sum(1).mean())
    return float(cifre_efective), float(ezitare)


def evalueaza(judecator, imagini):
    """
    imagini: (n, 784) in [-1, 1].
    Returneaza dict cu claritatea medie, distributia pe cifre, Inception Score
    si predictiile.
    """
    p = judecator.predict(imagini, batch_size=1000, verbose=0)
    cifre = p.argmax(1)
    is_medie, is_std = scor_inception(p)
    cifre_efective, ezitare = descompune_is(p)
    return {
        "claritate": float(p.max(1).mean()),
        "distributie": np.bincount(cifre, minlength=10) / len(cifre),
        "is": is_medie,
        "is_std": is_std,
        "cifre_efective": cifre_efective,
        "ezitare": ezitare,
        "p": p,
        "cifre": cifre,
        "incredere": p.max(1),
    }
