"""
Incarcarea si pregatirea setului MNIST pentru GAN.

Diferenta fata de Lab 2: acolo pixelii erau in [0, 1] pentru ca decoderul
se termina in sigmoid. Aici generatorul se termina in tanh, care scoate
valori in [-1, 1], deci datele reale trebuie aduse in acelasi interval.
Altfel discriminatorul ar putea separa realul de fals doar dupa semnul
pixelilor, fara sa invete nimic despre forma cifrelor.
"""

import numpy as np
import tensorflow as tf

DIM_IMAGINE = 784
LATENT_DIM = 100
BATCH_SIZE = 128


def incarca_mnist():
    """
    Returneaza (x_train, y_train).

    x_train: float32, forma (60000, 784), valori in [-1, 1].
    y_train: etichetele originale. NU intra in antrenarea GAN-ului, le
             pastram doar ca sa verificam mai tarziu ce cifre genereaza.
    """
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x = x_train.astype(np.float32)
    x = (x - 127.5) / 127.5          # 0 -> -1, 127.5 -> 0, 255 -> 1
    x = x.reshape(-1, DIM_IMAGINE)   # (60000, 28, 28) -> (60000, 784)
    return x, y_train


def creeaza_dataset(x, batch_size=BATCH_SIZE):
    """Amesteca imaginile si le grupeaza in loturi pentru antrenare."""
    return (
        tf.data.Dataset.from_tensor_slices(x)
        .shuffle(len(x))
        .batch(batch_size)
    )


def pentru_afisare(img):
    """Readuce imaginile din [-1, 1] in [0, 1], forma (n, 28, 28)."""
    return (np.asarray(img).reshape(-1, 28, 28) + 1) / 2
