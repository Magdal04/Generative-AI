"""
Salvarea greutatilor la epoci alese, pentru a putea compara starea
modelului in timp.

Antrenarea obisnuita pastreaza doar rezultatul final. Aici salvam
instantanee la epoci fixe, ca sa se poata vedea in interfata cum se
formeaza detectorii din primul strat: la inceput sunt zgomot din
initializarea aleatoare, apoi capata structura.

Nu e un callback din Keras pentru ca ModelCheckpoint salveaza dupa un
criteriu (cel mai bun scor) sau la fiecare epoca, nu la o lista de epoci
alese manual.
"""

from pathlib import Path

from tensorflow import keras

BASE_DIR = Path(__file__).parent.parent
DIR_CKPT = BASE_DIR / "checkpoints"


class SalveazaLaEpoci(keras.callbacks.Callback):
    """Salveaza greutatile exact la epocile din lista, in DIR_CKPT."""

    def __init__(self, epoci, prefix="ae"):
        super().__init__()
        self.epoci = set(epoci)
        self.prefix = prefix
        self.salvate = []
        DIR_CKPT.mkdir(exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Keras numeroteaza epocile de la 0; noi vorbim despre epoca 1, 2, ...
        nr = epoch + 1
        if nr in self.epoci:
            cale = DIR_CKPT / f"{self.prefix}_ep{nr:03d}.weights.h5"
            self.model.save_weights(cale)
            self.salvate.append(nr)
