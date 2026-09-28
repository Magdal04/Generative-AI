"""
Antrenarea VAE-ului pana la convergenta, cu acelasi criteriu ca AE clasic.

Criteriul ales la AE: min_delta = 0.0001 pe BCE medie per pixel (circa o
trei-suta parte din eroarea ramasa), patience = 10. Loss-ul VAE este suma
pe cei 784 de pixeli, deci acelasi criteriu relativ inseamna
min_delta = 0.0001 * 784. Fara scalare, VAE ar fi judecat de 784 de ori
mai strict decat AE si comparatia n-ar mai fi pe teren egal.

Rulare: python antreneaza_vae.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import numpy as np
from tensorflow import keras

from src.checkpointuri import SalveazaLaEpoci
from src.data import incarca_mnist
from src.vae import construieste_vae

MIN_DELTA = 0.0001 * 784
PATIENCE = 10
EPOCI_MAX = 300

keras.utils.set_random_seed(42)
x_train, x_test, y_train, y_test = incarca_mnist()

vae, enc, dec = construieste_vae(beta=1.0, reducere="suma")
vae.compile(optimizer="adam")

oprire = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=MIN_DELTA,
                                       patience=PATIENCE, restore_best_weights=True)
ckpt = SalveazaLaEpoci([1, 5, 20, 50, 130], prefix="vae")

t0 = time.time()
h = vae.fit(x_train, x_train, epochs=EPOCI_MAX, batch_size=256, shuffle=True,
            validation_split=0.1, callbacks=[oprire, ckpt], verbose=0)
secunde = time.time() - t0

(BASE_DIR / "modele").mkdir(exist_ok=True)
(BASE_DIR / "istoric").mkdir(exist_ok=True)
vae.save_weights(BASE_DIR / "modele" / "vae.weights.h5")
vae.save_weights(BASE_DIR / "checkpoints" / "vae_final.weights.h5")

ist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
best = int(np.argmin(ist["val_loss"])) + 1
with open(BASE_DIR / "istoric" / "vae.json", "w", encoding="utf-8") as f:
    json.dump({"istoric": ist, "secunde": secunde, "epoca_best": best,
               "epoci_rulate": len(ist["loss"]), "epoci_ckpt": ckpt.salvate,
               "min_delta": MIN_DELTA, "patience": PATIENCE, "beta": 1.0}, f, indent=2)

r = vae.evaluate(x_test, x_test, verbose=0, return_dict=True)
print(f"timp          : {secunde:.1f} s ({secunde/len(ist['loss']):.1f} s/epoca)")
print(f"epoci rulate  : {len(ist['loss'])}")
print(f"epoca best    : {best}")
print(f"checkpointuri : {ckpt.salvate}")
print(f"BCE/pixel test: {r['bce_pixel']:.4f}   (AE clasic: 0.0775, podea 0.0571)")
print(f"KL test       : {r['kl']:.2f}")
