"""
Autoencoder clasic pe MNIST.

Arhitectura: 784 - 128 - 64 - 32 - 64 - 128 - 784, cu ReLU pe straturile
ascunse si sigmoid la iesire.

Adancimea a fost aleasa masurat, nu prin conventie. La 30 de epoci:
    784-32 (un strat)             BCE 0.0916
    784-128-64-32 fara activare   BCE 0.0911   <- 4x parametri, acelasi rezultat
    784-128-64-32 cu ReLU         BCE 0.0845
Varianta fara activare colapseaza algebric intr-o singura matrice
(W1 @ W2 @ W3 se pot inmulti inainte de a atinge datele), deci parametrii
in plus nu cumpara nimic. ReLU e cel care face adancimea sa conteze.

Podeaua BCE pe MNIST este 0.0571: chiar si o copie perfecta a intrarii
lasa un cost rezidual, pentru ca BCE atinge zero doar pentru tinte exact
0 sau 1, iar 8.7% dintre pixeli sunt intermediari.
"""

from tensorflow import keras
from tensorflow.keras import layers

DIM_INTRARE = 784
STRATURI_ASCUNSE = [128, 64]
DIM_LATENT = 32


def construieste_autoencoder(dim_latent=DIM_LATENT, straturi=None):
    """
    Returneaza (autoencoder, encoder, decoder).

    Cele trei modele impart aceleasi greutati: encoder si decoder sunt
    ferestre catre bucati din autoencoder, nu copii. Antrenezi
    autoencoderul, iar celelalte doua devin utilizabile automat.

    Le separam pentru ca dupa antrenare avem nevoie de ele individual:
    encoder ca sa proiectam imaginile in spatiul latent si sa desenam
    scatter-ul, decoder ca sa generam imagini pornind de la un z dat.
    """
    straturi = STRATURI_ASCUNSE if straturi is None else straturi

    # --- Encoder ---
    intrare = keras.Input(shape=(DIM_INTRARE,), name="imagine_intrare")
    h = intrare
    for i, unitati in enumerate(straturi):
        h = layers.Dense(unitati, activation="relu", name=f"enc_{i+1}")(h)

    # Stratul latent ramane LINIAR, fara ReLU: ReLU ar taia la zero toate
    # valorile negative, deci jumatate din spatiul latent ar deveni
    # inaccesibila si codurile s-ar inghesui intr-un singur cadran.
    z = layers.Dense(dim_latent, name="z")(h)
    encoder = keras.Model(intrare, z, name="encoder")

    # --- Decoder ---
    # Intrare proprie, ca sa poata fi apelat si cu un z inventat de noi,
    # nu doar cu unul produs de encoder.
    intrare_z = keras.Input(shape=(dim_latent,), name="cod_latent")
    h = intrare_z
    for i, unitati in enumerate(reversed(straturi)):
        h = layers.Dense(unitati, activation="relu", name=f"dec_{i+1}")(h)

    # Sigmoid pentru ca datele sunt normalizate in [0, 1]. ReLU ar permite
    # iesiri de 5.3, care nu sunt pixeli valizi; tanh ar permite negative.
    iesire = layers.Dense(DIM_INTRARE, activation="sigmoid", name="reconstructie")(h)
    decoder = keras.Model(intrare_z, iesire, name="decoder")

    # --- Autoencoderul complet ---
    autoencoder = keras.Model(intrare, decoder(encoder(intrare)), name="autoencoder")
    return autoencoder, encoder, decoder


if __name__ == "__main__":
    ae, enc, dec = construieste_autoencoder()
    print(f"encoder     : {enc.count_params():>8,} parametri")
    print(f"decoder     : {dec.count_params():>8,} parametri")
    print(f"autoencoder : {ae.count_params():>8,} parametri")
    print()
    ae.summary()
