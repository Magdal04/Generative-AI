"""Functii de desen comune pentru raport si aplicatie."""

import matplotlib.pyplot as plt
import numpy as np

from src.data import pentru_afisare

C_REAL, C_FALS, C_BAZA, C_BUN, C_RAU = "#1565c0", "#ef6c00", "#9e9e9e", "#2e7d32", "#c62828"


def grila(imagini, randuri, coloane, ax=None, titlu=None, marime=1.0):
    """Deseneaza imagini (n, 784) in [-1, 1] ca o singura grila randuri x coloane."""
    img = pentru_afisare(imagini[: randuri * coloane])
    mozaic = np.ones((randuri * 29 - 1, coloane * 29 - 1))
    for i, im in enumerate(img):
        r, c = divmod(i, coloane)
        mozaic[r * 29 : r * 29 + 28, c * 29 : c * 29 + 28] = im
    if ax is None:
        _, ax = plt.subplots(figsize=(coloane * 0.6 * marime, randuri * 0.6 * marime))
    ax.imshow(mozaic, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    if titlu:
        ax.set_title(titlu, fontsize=10)
    return ax


def distante_la_cel_mai_apropiat(a, b, lot=250):
    """Pentru fiecare rand din a: distanta L2 minima si indexul celui mai apropiat rand din b."""
    b2 = (b ** 2).sum(1)
    dist, idx = [], []
    for i in range(0, len(a), lot):
        x = a[i : i + lot]
        d = (x ** 2).sum(1)[:, None] + b2[None, :] - 2 * x @ b.T
        idx.append(d.argmin(1))
        dist.append(np.sqrt(np.maximum(d.min(1), 0)))
    return np.concatenate(dist), np.concatenate(idx)
