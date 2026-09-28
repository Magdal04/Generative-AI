"""
Variational Autoencoder pe MNIST.

Aceeasi arhitectura ca autoencoderul clasic (784-128-64-...-784), cu doua
schimbari, si doar acestea doua:

1. Encoderul nu mai produce un punct, ci parametrii unei distributii:
   mu (centrul) si log_var (latimea). Codul z se ESANTIONEAZA din acea
   distributie, deci aceeasi imagine da un z usor diferit la fiecare
   trecere.

2. Loss-ul primeste un al doilea termen, KL, care impinge toate
   distributiile catre N(0,1).

Motivul, masurat pe autoencoderul clasic din acest laborator: un punct
tras la intamplare din spatiul latent al AE se afla la distanta 15.57 de
cel mai apropiat cod real, in timp ce codurile reale sunt la 7.92 unele
de altele. Raport 1.97 - tragi in gol. Intr-un spatiu N(0,1), acelasi
raport este 0.99: nu mai exista gol.

Beta = 1 este formularea originala, in care loss-ul are o derivare
matematica exacta. Alte valori sunt alegeri euristice care trebuie
justificate separat.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DIM_INTRARE = 784
STRATURI_ASCUNSE = [128, 64]
DIM_LATENT = 32


class Esantionare(layers.Layer):
    """
    Produce z = mu + sigma * eps, cu eps tras din N(0,1).

    Nu esantionam direct din N(mu, sigma) pentru ca esantionarea nu are
    derivata: nu poti calcula "cu cat se schimba numarul tras daca modific
    mu", deci gradientul nu ar putea ajunge la encoder. Trucul, numit
    reparametrizare, muta aleatorul intr-o variabila separata eps, care nu
    depinde de nimic invatabil. Drumul mu -> z ramane o simpla adunare, deci
    derivabil.

    Reteaua produce log_var, nu sigma, din doua motive: log_var poate lua
    orice valoare reala (sigma trebuie sa fie pozitiva, ceea ce ar cere o
    constrangere), iar exp(0.5 * log_var) e mereu pozitiv fara efort.
    """

    def call(self, intrari):
        mu, log_var = intrari
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * eps


def construieste_encoder(dim_latent=DIM_LATENT, straturi=None):
    """
    Returneaza un model care primeste o imagine si scoate [mu, log_var, z].

    Scoatem toate trei pentru ca avem nevoie de fiecare separat:
    mu si log_var intra in termenul KL al loss-ului, z merge la decoder.
    In plus, la evaluare folosim mu singur ca reprezentare determinista a
    imaginii, fara zgomotul esantionarii.
    """
    straturi = STRATURI_ASCUNSE if straturi is None else straturi

    intrare = keras.Input(shape=(DIM_INTRARE,), name="imagine_intrare")
    h = intrare
    for i, unitati in enumerate(straturi):
        h = layers.Dense(unitati, activation="relu", name=f"enc_{i+1}")(h)

    # Doua capete paralele din acelasi strat ascuns. Ambele liniare:
    # mu trebuie sa poata fi negativ, iar log_var la fel (log_var negativ
    # inseamna sigma sub 1, adica un nor ingust).
    mu = layers.Dense(dim_latent, name="mu")(h)
    log_var = layers.Dense(dim_latent, name="log_var")(h)
    z = Esantionare(name="z")([mu, log_var])

    return keras.Model(intrare, [mu, log_var, z], name="encoder_vae")


def construieste_decoder(dim_latent=DIM_LATENT, straturi=None):
    """Identic cu decoderul autoencoderului clasic. Aici nu se schimba nimic."""
    straturi = STRATURI_ASCUNSE if straturi is None else straturi
    intrare_z = keras.Input(shape=(dim_latent,), name="cod_latent")
    h = intrare_z
    for i, unitati in enumerate(reversed(straturi)):
        h = layers.Dense(unitati, activation="relu", name=f"dec_{i+1}")(h)
    iesire = layers.Dense(DIM_INTRARE, activation="sigmoid", name="reconstructie")(h)
    return keras.Model(intrare_z, iesire, name="decoder_vae")


class VAE(keras.Model):
    """
    Model cu bucla de antrenare proprie, pentru ca loss-ul are nevoie de
    mu si log_var, nu doar de iesire si tinta. model.compile(loss=...)
    primeste doar (tinta, iesire), deci nu poate calcula KL.

    loss = reconstructie + beta * KL

    reducere="suma"  : reconstructia e suma BCE pe cei 784 de pixeli.
                       Doar asa beta=1 corespunde formularii originale.
    reducere="medie" : media BCE pe pixeli, cum face Keras implicit.
                       Echivalent ascuns cu beta = 784 * beta, pentru ca
                       reconstructia devine de 784 de ori mai mica fata de KL.
    """

    def __init__(self, encoder, decoder, beta=1.0, reducere="suma", **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.reducere = reducere
        self.m_loss = keras.metrics.Mean(name="loss")
        # BCE medie pe pixel, calculata la fel ca la AE clasic, ca sa se
        # poata compara direct cu 0.0775 si cu podeaua 0.0571.
        self.m_bce = keras.metrics.Mean(name="bce_pixel")
        self.m_kl = keras.metrics.Mean(name="kl")

    @property
    def metrics(self):
        return [self.m_loss, self.m_bce, self.m_kl]

    def call(self, x, training=False):
        mu, log_var, z = self.encoder(x, training=training)
        # La antrenare decoderul primeste z esantionat. La evaluare
        # folosim mu, adica centrul norului, ca reconstructia sa fie
        # determinista si comparabila cu AE.
        return self.decoder(z if training else mu, training=training)

    def _loss(self, x, rec, mu, log_var):
        rec = tf.clip_by_value(rec, 1e-7, 1.0 - 1e-7)
        bce = -(x * tf.math.log(rec) + (1.0 - x) * tf.math.log(1.0 - rec))
        bce_pixel = tf.reduce_mean(bce)
        if self.reducere == "suma":
            reconstructie = tf.reduce_mean(tf.reduce_sum(bce, axis=1))
        else:
            reconstructie = tf.reduce_mean(tf.reduce_mean(bce, axis=1))
        # KL fata de N(0,1), forma inchisa pentru doua gaussiene.
        # Suma pe cele 32 de dimensiuni, media pe batch.
        kl = tf.reduce_mean(tf.reduce_sum(
            -0.5 * (1.0 + log_var - tf.square(mu) - tf.exp(log_var)), axis=1))
        return reconstructie + self.beta * kl, bce_pixel, kl

    def _actualizeaza(self, loss, bce_pixel, kl):
        self.m_loss.update_state(loss)
        self.m_bce.update_state(bce_pixel)
        self.m_kl.update_state(kl)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(x, training=True)
            rec = self.decoder(z, training=True)
            loss, bce_pixel, kl = self._loss(x, rec, mu, log_var)
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return self._actualizeaza(loss, bce_pixel, kl)

    def test_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        mu, log_var, z = self.encoder(x, training=False)
        rec = self.decoder(z, training=False)
        loss, bce_pixel, kl = self._loss(x, rec, mu, log_var)
        return self._actualizeaza(loss, bce_pixel, kl)


def construieste_vae(beta=1.0, reducere="suma", dim_latent=DIM_LATENT):
    """Returneaza (vae, encoder, decoder), la fel ca la autoencoderul clasic."""
    enc = construieste_encoder(dim_latent)
    dec = construieste_decoder(dim_latent)
    vae = VAE(enc, dec, beta=beta, reducere=reducere, name="vae")
    # Un model definit ca clasa e "construit" abia cand e apelat intreg,
    # vae(x). Bucla de antrenare apeleaza direct encoder si decoder, deci
    # fara acest apel VAE ramane marcat neconstruit si save_weights refuza.
    vae(tf.zeros((1, DIM_INTRARE)))
    return vae, enc, dec


if __name__ == "__main__":
    enc = construieste_encoder()
    enc.summary()
