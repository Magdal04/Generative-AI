"""
Antrenarea autoencoderelor, cu istoric salvat pentru comparatie.

Strategia aleasa: antrenam pana cand imbunatatirile devin ignorabile, nu
un numar fix de epoci. Criteriul de oprire e EarlyStopping cu min_delta.

Pragurile vin dintr-o masuratoare, nu din conventie. Dupa 30 de epoci,
AE clasic era la BCE 0.0857, adica 0.0286 peste podeaua de 0.0571, si
mai castiga ~0.0001 pe epoca. Am ales min_delta=0.0001, adica aproximativ
o trei-suta parte din eroarea ramasa: sub acest castig, o epoca de 1.6
secunde nu se mai justifica. patience=10 lasa modelul sa treaca peste
platouri temporare inainte de a declara convergenta.

EPOCI_MAX=300 e doar plafon de siguranta; oprirea reala vine de la
callback. restore_best_weights aduce inapoi greutatile de la minimul
validarii, nu pe cele din ultima epoca (care poate fi deja mai proasta).

Ambele modele, AE si VAE, primesc acelasi criteriu de oprire. Asta
pastreaza comparatia corecta: fiecare e judecat la convergenta proprie,
nu la un numar de epoci care ar putea avantaja arbitrar unul dintre ele.
"""

import json
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

BASE_DIR = Path(__file__).parent.parent
DIR_MODELE = BASE_DIR / "modele"
DIR_ISTORIC = BASE_DIR / "istoric"
EPOCI_MAX = 300
BATCH = 256
VALIDATION_SPLIT = 0.1
MIN_DELTA = 0.0001
PATIENCE = 10


def antreneaza(model, x_train, nume, epoci=EPOCI_MAX, batch=BATCH,
               min_delta=MIN_DELTA, patience=PATIENCE, verbose=0):
    """
    Antreneaza si returneaza (istoric_dict, secunde, epoca_cea_mai_buna).

    Tinta este chiar intrarea: model.fit(x, x). Nicio eticheta nu intra
    aici — de asta autoencoderul e self-supervised.

    Salveaza greutatile in modele/<nume>.weights.h5 si istoricul in
    istoric/<nume>.json, ca sa putem reface graficele fara sa reantrenam.
    """
    DIR_MODELE.mkdir(exist_ok=True)
    DIR_ISTORIC.mkdir(exist_ok=True)

    # min_delta e prag ABSOLUT pe val_loss, nu procent: o scadere sub
    # aceasta valoare nu se considera imbunatatire si incrementeaza wait.
    oprire = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=min_delta,
        patience=patience,
        restore_best_weights=True,
        verbose=verbose,
    )

    t0 = time.time()
    h = model.fit(
        x_train, x_train,
        epochs=epoci,
        batch_size=batch,
        shuffle=True,
        validation_split=VALIDATION_SPLIT,
        callbacks=[oprire],
        verbose=verbose,
    )
    secunde = time.time() - t0

    istoric = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    epoca_best = int(np.argmin(istoric["val_loss"])) + 1

    model.save_weights(DIR_MODELE / f"{nume}.weights.h5")
    with open(DIR_ISTORIC / f"{nume}.json", "w", encoding="utf-8") as f:
        json.dump(
            {"istoric": istoric, "secunde": secunde, "epoca_best": epoca_best,
             "epoci_rulate": len(istoric["loss"]), "epoci_max": epoci,
             "batch": batch, "min_delta": min_delta, "patience": patience},
            f, indent=2,
        )

    return istoric, secunde, epoca_best


def raport(nume, istoric, secunde, epoca_best):
    """Tipareste ce s-a intamplat la antrenare, inclusiv daca s-a supraantrenat."""
    tr = istoric["loss"]
    va = istoric["val_loss"]
    rulate = len(va)
    print(f"--- {nume} ---")
    print(f"timp              : {secunde:.1f} s ({secunde/rulate:.1f} s/epoca)")
    print(f"epoci rulate      : {rulate}")
    print(f"epoca cea mai buna: {epoca_best}")
    print(f"val_loss la best  : {va[epoca_best-1]:.4f}")
    print(f"val_loss la final : {va[-1]:.4f}")
    print(f"train_loss final  : {tr[-1]:.4f}")
    delta = va[-1] - va[epoca_best - 1]
    if rulate == EPOCI_MAX:
        print("a atins plafonul de epoci: NU s-a oprit singur, ar merita mai mult timp")
    elif delta > 0.0005:
        print(f"supraantrenare: val_loss a urcat cu {delta:.4f} dupa epoca {epoca_best}")
    else:
        print("platou: val_loss nu s-a mai imbunatatit semnificativ")
