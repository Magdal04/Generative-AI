"""
Comparatia AE clasic vs VAE, pe cele trei lucruri pe care le cere
laboratorul: reconstructie, spatiu latent, generare.

Rulare: python compara.py   (dupa ce ambele modele sunt antrenate)
Figurile ajung in figuri/.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.autoencoder import construieste_autoencoder
from src.data import incarca_mnist
from src.vae import construieste_vae

DIR_FIG = BASE_DIR / "figuri"
DIR_FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

x_train, x_test, y_train, y_test = incarca_mnist()

ae, ae_enc, ae_dec = construieste_autoencoder()
ae.load_weights(BASE_DIR / "modele" / "ae_clasic.weights.h5")

vae, vae_enc, vae_dec = construieste_vae()
vae(x_test[:1])  # construieste greutatile inainte de incarcare
vae.load_weights(BASE_DIR / "modele" / "vae.weights.h5")


def cod_ae(x):
    return ae_enc.predict(x, verbose=0)


def cod_vae(x):
    # mu = centrul norului, reprezentarea determinista a imaginii
    return vae_enc.predict(x, verbose=0)[0]


# ---------------------------------------------------------------- 1. reconstructie
idx = [int(np.where(y_test == c)[0][0]) for c in range(10)]
orig = x_test[idx]
rec_ae = ae_dec.predict(cod_ae(orig), verbose=0)
rec_vae = vae_dec.predict(cod_vae(orig), verbose=0)

fig, axe = plt.subplots(3, 10, figsize=(10, 3.4))
for c in range(10):
    for r, (im, nume) in enumerate([(orig, "original"), (rec_ae, "AE"), (rec_vae, "VAE")]):
        axe[r, c].imshow(im[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axe[r, c].axis("off")
    axe[0, c].set_title(str(c), fontsize=9)
for r, nume in enumerate(["original", "AE", "VAE"]):
    axe[r, 0].text(-8, 14, nume, fontsize=9, ha="right", va="center")
fig.suptitle("Reconstructie: aceleasi imagini de test", fontsize=11)
fig.savefig(DIR_FIG / "comparatie_reconstructie.png", dpi=120, bbox_inches="tight")
plt.close(fig)

bce = lambda x, p: float(np.mean(-(x * np.log(np.clip(p, 1e-7, 1)) +
                                   (1 - x) * np.log(np.clip(1 - p, 1e-7, 1)))))
bce_ae = bce(x_test, ae_dec.predict(cod_ae(x_test), verbose=0))
bce_vae = bce(x_test, vae_dec.predict(cod_vae(x_test), verbose=0))

# ---------------------------------------------------------------- 2. generare
# Acelasi test ca la AE: z tras la noroc, fara nicio imagine in spate.
N = 10
Z_ae = cod_ae(x_test)
gen = {
    "AE, z din N(0,1)": ae_dec.predict(rng.normal(0, 1, (N, 32)), verbose=0),
    "AE, z cu media/std reale": ae_dec.predict(
        rng.normal(Z_ae.mean(0), Z_ae.std(0), (N, 32)), verbose=0),
    "VAE, z din N(0,1)": vae_dec.predict(rng.normal(0, 1, (N, 32)), verbose=0),
}
fig, axe = plt.subplots(len(gen), N, figsize=(N, len(gen) * 1.2))
for r, (nume, im) in enumerate(gen.items()):
    for c in range(N):
        axe[r, c].imshow(im[c].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axe[r, c].axis("off")
    axe[r, 0].set_title(nume, fontsize=8, loc="left", x=-0.05)
fig.suptitle("Generare: decoderul primeste z inventat, nicio imagine reala", fontsize=11)
fig.savefig(DIR_FIG / "comparatie_generare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. golul din latent
# Raportul masurat la AE: cat de departe cade un z tras la noroc de cel mai
# apropiat cod real, fata de cat de departe sunt codurile reale intre ele.
def raport_gol(Z, probe):
    real = Z[rng.choice(len(Z), 300, replace=False)]
    d_real = [np.sort(np.linalg.norm(Z - p, axis=1))[1] for p in real]
    d_probe = [np.linalg.norm(Z - p, axis=1).min() for p in probe]
    return float(np.mean(d_probe) / np.mean(d_real))

# La VAE masuram pe z esantionat, nu pe mu. Norii sunt cei care umplu
# golul; centrele lor (mu) sunt tot puncte izolate. Masurat pe mu, raportul
# iesea 2.84 si contrazicea figura de generare.
Z_vae = vae_enc.predict(x_test, verbose=0)[2]
gol_ae = raport_gol(Z_ae, rng.normal(Z_ae.mean(0), Z_ae.std(0), (300, 32)))
gol_vae = raport_gol(Z_vae, rng.normal(0, 1, (300, 32)))

print(f"{'':<28}{'AE clasic':>12}{'VAE':>12}")
print("-" * 52)
print(f"{'BCE/pixel pe test':<28}{bce_ae:>12.4f}{bce_vae:>12.4f}")
print(f"{'raport gol (1.0 = fara gol)':<28}{gol_ae:>12.2f}{gol_vae:>12.2f}")
print("\nfiguri: comparatie_reconstructie.png, comparatie_generare.png")
