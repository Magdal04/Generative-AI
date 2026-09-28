"""
Antreneaza GAN-ul si salveaza tot ce folosesc aplicatia si raportul:
  checkpoints/G_epXXX, D_epXXX   ponderile la epocile alese
  istoric/istoric_gan.json       media pe epoca a loss-urilor si a lui p_real/p_fake
  istoric/z_fix.npy + evolutie.npy   imaginile din zgomotul fix, dupa fiecare epoca
"""

import json
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

from src.antrenare import antreneaza
from src.data import LATENT_DIM, creeaza_dataset, incarca_mnist
from src.modele import construieste_discriminator, construieste_generator

EPOCI = 100
EPOCI_CHECKPOINT = (1, 2, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100)

def main():
    tf.random.set_seed(42)
    os.makedirs("istoric", exist_ok=True)
    x, _ = incarca_mnist()
    G, D = construieste_generator(), construieste_discriminator()
    z_fix = tf.random.normal((16, LATENT_DIM), seed=7)
    np.save("istoric/z_fix.npy", z_fix.numpy())

    evolutie = []
    def la_fiecare_epoca(epoca, imagini):
        evolutie.append(imagini)
        np.save("istoric/evolutie.npy", np.stack(evolutie))

    istoric = antreneaza(G, D, creeaza_dataset(x), EPOCI, EPOCI_CHECKPOINT,
                         z_fix, la_fiecare_epoca)
    with open("istoric/istoric_gan.json", "w") as f:
        json.dump(istoric, f)


if __name__ == "__main__":
    main()
