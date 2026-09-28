"""
Generatorul si discriminatorul GAN-ului, straturi Dense (Goodfellow, 2014).

Generatorul transforma 100 de numere aleatoare intr-o imagine de 784 pixeli.
Ultimul strat e tanh ca sa produca exact intervalul datelor reale, [-1, 1].
"""

import tensorflow as tf
from tensorflow.keras import layers

from src.data import DIM_IMAGINE, LATENT_DIM


def construieste_generator(latent_dim=LATENT_DIM):
    """z (latent_dim) -> 256 -> 512 -> 784 pixeli in [-1, 1]."""
    return tf.keras.Sequential(
        [
            layers.Input((latent_dim,)),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(DIM_IMAGINE, activation="tanh"),
        ],
        name="generator",
    )


def construieste_discriminator():
    """784 pixeli -> 512 -> 256 -> P(real) intre 0 si 1."""
    return tf.keras.Sequential(
        [
            layers.Input((DIM_IMAGINE,)),
            layers.Dense(512),
            layers.LeakyReLU(0.2),
            layers.Dense(256),
            layers.LeakyReLU(0.2),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )
