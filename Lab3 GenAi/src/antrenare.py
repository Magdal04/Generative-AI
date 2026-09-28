"""
Pasul de antrenare GAN: un lot de imagini reale -> o actualizare pentru D,
apoi o actualizare pentru G.

Ordinea conteaza: D invata intai sa separe realul de falsurile curente,
apoi G primeste feedback de la acest D proaspat actualizat.
"""

import tensorflow as tf

from src.data import LATENT_DIM


def creeaza_optimizatoare():
    """Cate un Adam separat pentru fiecare retea, ca in cerinta."""
    return (
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),  # pentru G
        tf.keras.optimizers.Adam(2e-4, beta_1=0.5),  # pentru D
    )


def creeaza_pas_antrenare(G, D, opt_g, opt_d, latent_dim=LATENT_DIM):
    bce = tf.keras.losses.BinaryCrossentropy()

    # tf.function compileaza pasul intr-un graf o singura data; fara el,
    # Python ar interpreta fiecare operatie la fiecare lot (mult mai lent).
    @tf.function
    def pas(real):
        # Varianta B: atatea falsuri cate imagini reale are lotul
        # (ultimul lot are 96, nu 128).
        n = tf.shape(real)[0]

        # --- 1. Discriminatorul: real = 1, fals = 0 ---
        z = tf.random.normal((n, latent_dim))
        fake = G(z, training=True)
        with tf.GradientTape() as tape:
            p_real = D(real, training=True)
            p_fake = D(fake, training=True)
            d_loss = bce(tf.ones_like(p_real), p_real) + bce(
                tf.zeros_like(p_fake), p_fake
            )
        grad_d = tape.gradient(d_loss, D.trainable_variables)
        opt_d.apply_gradients(zip(grad_d, D.trainable_variables))
        #                       ^ doar ponderile lui D se misca aici

        # --- 2. Generatorul: vrea ca D sa spuna "real" (1) la falsuri ---
        z = tf.random.normal((n, latent_dim))  # zgomot nou
        with tf.GradientTape() as tape:
            p_fake_g = D(G(z, training=True), training=True)
            g_loss = bce(tf.ones_like(p_fake_g), p_fake_g)
        grad_g = tape.gradient(g_loss, G.trainable_variables)
        opt_g.apply_gradients(zip(grad_g, G.trainable_variables))
        #                       ^ doar ponderile lui G se misca: asta e
        #                         "D inghetat" -- gradientul trece prin D,
        #                         dar nu il modifica

        return {
            "d_loss": d_loss,
            "g_loss": g_loss,
            "p_real": tf.reduce_mean(p_real),
            "p_fake": tf.reduce_mean(p_fake),
        }

    return pas


def antreneaza(G, D, dataset, epoci, epoci_checkpoint=(), z_fix=None,
               la_fiecare_epoca=None, dir_checkpoint="checkpoints"):
    """
    Bucla completa de antrenare.

    epoci_checkpoint: epocile la care salvam ponderile lui G si D.
    z_fix:            zgomot fix; la_fiecare_epoca(epoca, G(z_fix)) e apelat
                      dupa fiecare epoca, ca sa desenam evolutia pe acelasi z.
    Returneaza istoricul: media pe epoca pentru d_loss, g_loss, p_real, p_fake.
    """
    import os, time
    import numpy as np

    opt_g, opt_d = creeaza_optimizatoare()
    pas = creeaza_pas_antrenare(G, D, opt_g, opt_d)
    os.makedirs(dir_checkpoint, exist_ok=True)
    istoric = {"d_loss": [], "g_loss": [], "p_real": [], "p_fake": []}

    for epoca in range(1, epoci + 1):
        t0 = time.time()
        sume = {k: [] for k in istoric}
        for lot in dataset:
            out = pas(lot)
            for k in istoric:
                sume[k].append(float(out[k]))
        for k in istoric:
            istoric[k].append(float(np.mean(sume[k])))

        if epoca in epoci_checkpoint:
            G.save_weights(f"{dir_checkpoint}/G_ep{epoca:03d}.weights.h5")
            D.save_weights(f"{dir_checkpoint}/D_ep{epoca:03d}.weights.h5")
        if la_fiecare_epoca is not None and z_fix is not None:
            la_fiecare_epoca(epoca, G(z_fix, training=False).numpy())

        print(f"epoca {epoca:3d} | d_loss {istoric['d_loss'][-1]:.3f} "
              f"g_loss {istoric['g_loss'][-1]:.3f} | p_real {istoric['p_real'][-1]:.3f} "
              f"p_fake {istoric['p_fake'][-1]:.3f} | {time.time() - t0:.1f}s", flush=True)
    return istoric
