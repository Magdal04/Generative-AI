"""
Incarcarea si pregatirea setului MNIST pentru autoencodere.

MNIST vine ca uint8 in intervalul 0..255. Retelele au nevoie de float32
in interval mic, altfel activarile satureaza si gradientul dispare.
Normalizam in [0, 1] pentru ca ultimul strat al decoderului e sigmoid,
care produce exact acest interval, si pentru ca binary crossentropy
interpreteaza fiecare pixel ca probabilitate de a fi aprins.
"""

import numpy as np
from tensorflow import keras


def incarca_mnist(aplatizat=True):
    """
    Returneaza (x_train, x_test, y_train, y_test).

    x_* sunt float32 in [0, 1].
    y_* raman etichetele originale: NU se folosesc la antrenare, doar
    pentru a colora punctele in graficul spatiului latent.

    aplatizat=True  -> forma (n, 784), pentru straturi Dense
    aplatizat=False -> forma (n, 28, 28, 1), pentru straturi convolutionale
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Conversia la float32 trebuie facuta INAINTE de impartire. Daca imparti
    # direct un uint8, numpy promoveaza oricum, dar a fi explicit evita
    # surprize la overflow in alte operatii pe acelasi array.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    if aplatizat:
        x_train = x_train.reshape(len(x_train), 784)
        x_test = x_test.reshape(len(x_test), 784)
    else:
        x_train = x_train.reshape(len(x_train), 28, 28, 1)
        x_test = x_test.reshape(len(x_test), 28, 28, 1)

    return x_train, x_test, y_train, y_test


def rezumat(x_train, x_test, y_train, y_test):
    """Tipareste ce s-a incarcat, ca sa se vada in raport cifrele reale."""
    print(f"x_train : {x_train.shape}  dtype={x_train.dtype}")
    print(f"x_test  : {x_test.shape}  dtype={x_test.dtype}")
    print(f"interval: [{x_train.min():.3f}, {x_train.max():.3f}]")
    print(f"medie   : {x_train.mean():.4f}")
    print(f"pixeli sub 0.1 : {(x_train < 0.1).mean() * 100:.1f}%")
    print(f"pixeli peste 0.9: {(x_train > 0.9).mean() * 100:.1f}%")
    print(f"pixeli intre    : {((x_train >= 0.1) & (x_train <= 0.9)).mean() * 100:.1f}%")
    print(f"distributie clase (train): {np.bincount(y_train)}")


if __name__ == "__main__":
    xtr, xte, ytr, yte = incarca_mnist()
    rezumat(xtr, xte, ytr, yte)
