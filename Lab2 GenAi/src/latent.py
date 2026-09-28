"""
Proiectia spatiului latent in 2D, pentru grafice: raport si aplicatie.

Spatiul latent are 32 de dimensiuni. PCA alege cele doua directii de-a
lungul carora codurile variaza cel mai mult si proiecteaza pe ele. E o
umbra a spatiului real: puncte departate in 32D pot cadea una peste alta
in 2D. De asta raportam mereu cat din variatie pastreaza proiectia.

Norul unei imagini VAE e o gaussiana cu latimi sigma pe fiecare axa. Proiectat
in 2D, ramane o gaussiana, deci se deseneaza ca elipsa.
"""

import numpy as np


class ProiectiePCA:
    """PCA cu numpy: potrivit o data pe codurile reale, apoi aplicat oricarui punct."""

    def __init__(self, Z):
        self.medie = Z.mean(axis=0)
        # SVD pe datele centrate: randurile lui Vt sunt directiile principale,
        # ordonate dupa cata variatie explica.
        _, s, Vt = np.linalg.svd(Z - self.medie, full_matrices=False)
        self.W = Vt[:2]                                   # (2, dim_latent)
        var = s ** 2
        self.varianta_pastrata = float(var[:2].sum() / var.sum())

    def aplica(self, Z):
        """(n, dim_latent) -> (n, 2)"""
        return (np.atleast_2d(Z) - self.medie) @ self.W.T

    def elipsa(self, sigma, k=2.0):
        """
        Elipsa norului N(mu, diag(sigma^2)) dupa proiectie, la k deviatii.
        Returneaza (latime, inaltime, unghi_grade) pentru matplotlib Ellipse.

        Covarianta proiectata e W diag(sigma^2) W^T, o matrice 2x2. Vectorii ei
        proprii dau orientarea elipsei, valorile proprii dau lungimea axelor.
        """
        C = self.W @ np.diag(np.asarray(sigma) ** 2) @ self.W.T
        val, vec = np.linalg.eigh(C)                      # crescator
        val = np.clip(val, 0, None)
        unghi = float(np.degrees(np.arctan2(vec[1, 1], vec[0, 1])))
        return 2 * k * float(np.sqrt(val[1])), 2 * k * float(np.sqrt(val[0])), unghi


def centre_cifre(P, y):
    """Mediana pozitiei 2D pentru fiecare cifra: unde punem eticheta pe harta."""
    return {int(c): np.median(P[y == c], axis=0) for c in np.unique(y)}


def pozitii_etichete(P, y, fractie=0.07):
    """
    Centrele cifrelor, departate cat sa nu se acopere etichetele.

    Unele cifre (3, 5, 8) au centrele aproape una de alta si etichetele s-ar
    suprapune. Le impingem in directia dintre ele pana ajung la o distanta
    minima, o fractie din intinderea hartii.
    """
    poz = {c: np.array(v, float) for c, v in centre_cifre(P, y).items()}
    d_min = fractie * np.ptp(P, axis=0).max()
    for _ in range(50):
        for a in poz:
            for b in poz:
                if a < b:
                    d = poz[b] - poz[a]
                    n = np.linalg.norm(d)
                    if n < d_min:
                        u = d / n if n > 1e-9 else np.array([1.0, 0.0])
                        poz[a] -= u * (d_min - n) / 2
                        poz[b] += u * (d_min - n) / 2
    return poz
