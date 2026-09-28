"""
Grafice pentru raport: reconstructii, spatiu latent, curbe de antrenare.

Toate functiile primesc date deja calculate si doar deseneaza. Nimic nu
se antreneaza aici, ca sa poata fi rulate de mai multe ori fara cost.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend fara fereastra, necesar la rulare din script
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent.parent
DIR_FIG = BASE_DIR / "figuri"


def _salveaza(fig, nume):
    DIR_FIG.mkdir(exist_ok=True)
    cale = DIR_FIG / nume
    fig.savefig(cale, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return cale


def reconstructii(originale, reconstruite, etichete=None, n=10, titlu="", nume="recon.png"):
    """
    Doua randuri: sus imaginile reale, jos ce a produs modelul.
    Sub fiecare pereche, eroarea absoluta medie pe pixel, ca sa se vada
    care cifre sunt grele pentru model, nu doar ca "arata bine".
    """
    fig, axe = plt.subplots(3, n, figsize=(n * 1.1, 3.6))
    for i in range(n):
        axe[0, i].imshow(originale[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axe[1, i].imshow(reconstruite[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1)

        # Harta de eroare: unde anume a gresit modelul.
        dif = np.abs(originale[i] - reconstruite[i]).reshape(28, 28)
        axe[2, i].imshow(dif, cmap="hot", vmin=0, vmax=1)

        for r in range(3):
            axe[r, i].axis("off")
        et = f"{etichete[i]}" if etichete is not None else ""
        axe[0, i].set_title(et, fontsize=9)
        axe[2, i].set_title(f"{dif.mean():.3f}", fontsize=7, color="darkred")

    axe[0, 0].set_ylabel("original", fontsize=8)
    axe[1, 0].set_ylabel("reconstruit", fontsize=8)
    axe[2, 0].set_ylabel("eroare", fontsize=8)
    fig.suptitle(titlu, fontsize=11)
    return _salveaza(fig, nume)


def curba_antrenare(istoric, epoca_best, podea=0.0571, titlu="", nume="curba.png"):
    """
    Doua panouri: stanga loss brut, dreapta loss peste podea pe scara log.
    Panoul din dreapta e cel informativ: arata cat din eroarea REALA a mai
    ramas, nu cat din numarul absolut care contine si podeaua intrinseca.
    """
    tr, va = istoric["loss"], istoric["val_loss"]
    ep = np.arange(1, len(tr) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))

    a1.plot(ep, tr, label="train")
    a1.plot(ep, va, label="validare")
    a1.axhline(podea, color="gray", ls="--", lw=1, label=f"podea BCE {podea}")
    a1.axvline(epoca_best, color="green", ls=":", lw=1, label=f"best: ep {epoca_best}")
    a1.set_xlabel("epoca"); a1.set_ylabel("BCE"); a1.legend(fontsize=8)
    a1.set_title("loss brut")

    a2.semilogy(ep, np.array(tr) - podea, label="train")
    a2.semilogy(ep, np.array(va) - podea, label="validare")
    a2.axvline(epoca_best, color="green", ls=":", lw=1)
    a2.set_xlabel("epoca"); a2.set_ylabel("BCE - podea (log)"); a2.legend(fontsize=8)
    a2.set_title("eroare reala ramasa")

    fig.suptitle(titlu, fontsize=11)
    return _salveaza(fig, nume)
