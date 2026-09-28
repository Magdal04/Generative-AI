"""Ruleaza judecatorul pe 10 000 de cifre generate la fiecare checkpoint al lui G."""

import glob
import json
import os
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf

from src.data import LATENT_DIM
from src.judecator import evalueaza, incarca_judecator
from src.modele import construieste_generator

def main():
    J, G = incarca_judecator(), construieste_generator()
    z = tf.random.normal((10000, LATENT_DIM), seed=123)
    rezultate = {}
    for cale in sorted(glob.glob("checkpoints/G_ep*.weights.h5")):
        ep = int(re.search(r"ep(\d+)", cale).group(1))
        G.load_weights(cale)
        r = evalueaza(J, G(z, training=False).numpy())
        rezultate[ep] = {"claritate": r["claritate"], "distributie": r["distributie"].tolist(),
                         "is": r["is"], "is_std": r["is_std"],
                         "cifre_efective": r["cifre_efective"], "ezitare": r["ezitare"]}
        print(f"epoca {ep:3d} | claritate {r['claritate']:.3f} | IS {r['is']:.2f} "
              f"(cifre efective {r['cifre_efective']:.2f} / ezitare {r['ezitare']:.2f}) | "
              f"min {r['distributie'].min():.3f} max {r['distributie'].max():.3f} | "
              + " ".join(f"{d}:{p:.2f}" for d, p in enumerate(r["distributie"])))
    with open("istoric/evaluare_judecator.json", "w") as f:
        json.dump(rezultate, f)


if __name__ == "__main__":
    main()
