"""
Interfata interactiva: desenezi o cifra si vezi ce se intampla in interiorul
autoencoderului, strat cu strat.

Rulare:  python app_desen.py

Ce arata:

1. Filtrele primului strat: greutatile fiecarui neuron din enc_1, desenate
   ca imagine 28x28. Fiecare neuron are exact 784 de greutati, cate una per
   pixel, deci se pot aseza inapoi in forma imaginii. Asta arata CE cauta
   neuronul, nu doar cat s-a aprins. Doar primul strat se poate vizualiza
   asa: de la al doilea incolo, intrarea nu mai e formata din pixeli.

2. Activarile pe drumul 784 -> 128 -> 64 -> 32 -> 64 -> 128 -> 784. Aici
   pozitia unui patratel in grila NU inseamna nimic: e doar neuronul 0, 1,
   2, ... asezat pe randuri ca sa incapa. Panoul arata CAT s-a aprins
   fiecare, nu unde.

Selectorul de epoci incarca greutatile salvate in timpul antrenarii. La
epoca 1 filtrele sunt inca aproape zgomotul din initializarea aleatoare;
pe parcurs capata structura. Comparatia intre epoci e argumentul ca
detectorii sunt invatati, nu prezenti de la inceput.

Comutatorul "preprocesare MNIST" arata de ce conteaza potrivirea cu
distributia de antrenare: acelasi desen, cu si fara centrare pe centrul de
masa, produce reconstructii vizibil diferite.
"""

import os
import sys
import tkinter as tk
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from PIL import Image, ImageDraw, ImageTk

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse

from src.autoencoder import construieste_autoencoder
from src.data import incarca_mnist
from src.latent import ProiectiePCA, pozitii_etichete
from src.preprocesare import desen_brut, desen_ca_mnist
from src.vae import construieste_vae

# Culorile modelelor pe harta latenta (varianta pentru fundal inchis a
# paletei validate; AE si VAE trec verificarea de daltonism).
CULOARE = {"AE": "#3987e5", "VAE": "#d95926"}
N_HARTA = 3000       # imagini de test desenate ca fundal gri pe harta
MAX_ISTORIC = 12     # cate desene/generari raman pe harta

# Ce difera intre cele doua modele, ca restul aplicatiei sa le trateze la fel.
# "latent" e stratul afisat in panoul z: la AE e z insusi, la VAE e mu,
# centrul norului, pentru ca z esantionat s-ar schimba la fiecare rulare.
MODELE = {
    "AE": {"construieste": construieste_autoencoder, "prefix": "ae",
           "istoric": "ae_clasic.json", "final": "ae_clasic.weights.h5",
           "latent": "z"},
    "VAE": {"construieste": construieste_vae, "prefix": "vae",
            "istoric": "vae.json", "final": "vae.weights.h5",
            "latent": "mu"},
}

PANZA = 280          # latura zonei de desen, in pixeli de ecran
GROSIME = 18         # grosimea pensulei; calibrata ca raportul trasatura/cadru
                     # sa semene cu MNIST dupa scalarea la 20x20
FUNDAL = "#1e1e1e"
TEXT = "#e0e0e0"


class Aplicatie:
    def __init__(self, radacina):
        self.radacina = radacina
        radacina.title("Autoencoder MNIST - ce se intampla in interior")
        radacina.configure(bg=FUNDAL)

        self._incarca_model()

        # Fundalul hartii latente: imagini de test reale, cu etichetele lor.
        # Etichetele nu intra in model, servesc doar ca harta sa aiba repere.
        _, x_test, _, y_test = incarca_mnist()
        self.x_harta, self.y_harta = x_test[:N_HARTA], y_test[:N_HARTA]

        # Istoricul pastreaza IMAGINILE desenate, nu coordonatele lor. La o
        # alta epoca sau alt model, aceleasi desene se reproiecteaza, deci se
        # vede cum se muta acelasi desen in spatiul latent.
        self.istoric = []
        self.id_desen = 0

        # Imaginea in care desenam efectiv. Canvas-ul Tkinter e doar afisaj;
        # pastram separat un array pe care il putem da modelului.
        self.desen = Image.new("L", (PANZA, PANZA), 0)
        self.creion = ImageDraw.Draw(self.desen)
        self.ultim = None

        self._construieste_interfata()
        self._deseneaza_filtre()
        self._reface_harta()
        self._deseneaza_harta()

    # ------------------------------------------------------------------
    def _gaseste_checkpointuri(self, cfg):
        """
        Returneaza ([(eticheta, cale), ...], info_final) pentru un model.
        Daca nu exista checkpointuri intermediare, ramane doar modelul final.
        """
        gasite = []
        dir_ckpt = BASE_DIR / "checkpoints"
        prefix = cfg["prefix"]
        if dir_ckpt.exists():
            for c in sorted(dir_ckpt.glob(prefix + "_ep*.weights.h5")):
                nr = int(c.stem.replace(prefix + "_ep", "").replace(".weights", ""))
                gasite.append(("epoca " + str(nr), c))
        final = BASE_DIR / "modele" / cfg["final"]
        info = ""
        if final.exists():
            # Citim epoca din istoric in loc sa o scriem in cod: dupa o
            # reantrenare cu alta samanta sau alt prag, numarul se schimba,
            # iar o eticheta fixa ar minti.
            eticheta, info = self._eticheta_final(cfg["istoric"])
            gasite.append((eticheta, final))
        return gasite, info

    def _eticheta_final(self, fisier):
        """('final (ep 130)', 'greutati de la ...') din istoric, daca exista."""
        cale = BASE_DIR / "istoric" / fisier
        try:
            import json
            with open(cale, encoding="utf-8") as f:
                d = json.load(f)
            info = "greutati de la epoca {} (cea mai buna), din {} rulate".format(
                d["epoca_best"], d["epoci_rulate"])
            return "final (ep {})".format(d["epoca_best"]), info
        except (ValueError, KeyError, OSError):
            return "final", ""

    def _incarca_model(self):
        """
        Construieste ambele modele o singura data. Comutarea intre ele
        schimba doar care e activ, fara reincarcare de la zero.
        """
        from tensorflow import keras
        self.modele = {}
        for nume, cfg in MODELE.items():
            ckpt, info = self._gaseste_checkpointuri(cfg)
            if not ckpt:
                continue
            complet, enc, dec = cfg["construieste"]()
            complet.load_weights(ckpt[-1][1])

            # Model care scoate activarile INTERMEDIARE, nu doar rezultatul
            # final. Fara asta am vedea doar intrarea si iesirea, adica exact
            # partea care nu explica nimic.
            m_enc = keras.Model(enc.input, [enc.get_layer(n).output
                                            for n in ("enc_1", "enc_2", cfg["latent"])])
            m_dec = keras.Model(dec.input, [dec.get_layer(n).output
                                            for n in ("dec_1", "dec_2", "reconstructie")])
            self.modele[nume] = {"complet": complet, "encoder": enc, "decoder": dec,
                                 "m_enc": m_enc, "m_dec": m_dec,
                                 "checkpointuri": ckpt, "info": info,
                                 "dim_latent": dec.input.shape[-1]}
        if not self.modele:
            raise SystemExit("Lipsesc greutatile in modele/. Antreneaza intai modelele.")
        self.activ = next(iter(self.modele))

    def _m(self):
        return self.modele[self.activ]

    def schimba_model(self):
        self.activ = self.model_sel.get()
        self._reface_selector_epoci()
        self._deseneaza_filtre()
        self._reface_harta()
        self.ruleaza(adauga=False)
        self._deseneaza_harta()

    def schimba_epoca(self, eticheta):
        """Incarca alt set de greutati si redeseneaza tot."""
        m = self._m()
        self.epoca_sel.set(eticheta)    # titlul hartii citeste de aici
        for et, cale in m["checkpointuri"]:
            if et == eticheta:
                m["complet"].load_weights(cale)
                text = m["info"] if et.startswith("final") else "greutati intermediare: " + et
                self.et_info_ep.config(text=self.activ + ": " + text)
                break
        self._deseneaza_filtre()
        self._reface_harta()
        self.ruleaza(adauga=False)
        self._deseneaza_harta()

    def _reface_selector_epoci(self):
        """Fiecare model are propriile checkpointuri, deci butoanele se refac."""
        for w in self.rand_epoci.winfo_children():
            w.destroy()
        m = self._m()
        # Revenim la greutatile finale ale modelului activat.
        m["complet"].load_weights(m["checkpointuri"][-1][1])
        self.epoca_sel.set(m["checkpointuri"][-1][0])
        tk.Label(self.rand_epoci, text="greutati de la:", bg=FUNDAL, fg=TEXT,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 6))
        for et, _ in m["checkpointuri"]:
            tk.Radiobutton(self.rand_epoci, text=et, variable=self.epoca_sel, value=et,
                           command=lambda: self.schimba_epoca(self.epoca_sel.get()),
                           bg=FUNDAL, fg=TEXT, selectcolor="#333",
                           activebackground=FUNDAL, activeforeground=TEXT,
                           font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
        self.et_info_ep.config(text=self.activ + ": " + m["info"])

    def genereaza(self):
        """
        Sare peste encoder: z tras din N(0,1), direct in decoder. Exact testul
        din compara.py. La AE iese mazgaleala, la VAE ar trebui sa iasa o
        cifra, pentru ca doar VAE a fost fortat sa umple N(0,1).
        """
        m = self._m()
        z = np.random.normal(0, 1, (1, m["dim_latent"])).astype("float32")
        d1, d2, rec = [np.asarray(v)[0] for v in m["m_dec"].predict(z, verbose=0)]
        gol = np.zeros((28, 28), dtype="float32")
        self._pune_imagine(self.et_intrare, gol)
        self._pune_imagine(self.et_iesire, rec.reshape(28, 28))
        self._pune_imagine(self.et_eroare, gol)
        for nume in ("enc_1", "enc_2"):
            self._pune_activari(nume, np.zeros(128 if nume == "enc_1" else 64, "float32"))
        for nume, vect in [("z", z[0]), ("dec_1", d1), ("dec_2", d2)]:
            self._pune_activari(nume, vect)
        self.stare.config(text="{}: GENERARE, fara nicio imagine de intrare\n"
                               "z tras din N(0,1), direct in decoder".format(self.activ))
        # Un z generat apartine unui singur spatiu latent, deci apare doar pe
        # harta modelului care l-a produs.
        self._adauga_istoric({"tip": "generat", "z": z[0], "model": self.activ})
        self.id_desen += 1

    # ------------------------------------------------------------------ harta
    def _latent(self, x):
        """(coduri, sigma sau None) pentru imagini, cu modelul activ."""
        out = self._m()["encoder"].predict(x, verbose=0, batch_size=512)
        if isinstance(out, list):                    # VAE: [mu, log_var, z]
            return out[0], np.exp(0.5 * out[1])
        return out, None                             # AE: z

    def _reface_harta(self):
        """
        Recalculeaza proiectia pentru modelul si epoca active. Se apeleaza
        la fiecare schimbare de greutati: alta epoca inseamna alt spatiu latent.
        """
        Z, _ = self._latent(self.x_harta)
        pca = ProiectiePCA(Z)
        P = pca.aplica(Z)
        self.harta = {"pca": pca, "P": P, "etichete": pozitii_etichete(P, self.y_harta)}

    def _adauga_istoric(self, intrare):
        """
        Desenul curent inlocuieste ultima lui versiune, in loc sa adauge un
        punct nou la fiecare trasatura. Asa punctul se misca pe harta in timp
        ce desenezi, iar un desen nou incepe dupa Sterge.
        """
        ultim = self.istoric[-1] if self.istoric else None
        if (intrare["tip"] == "desen" and ultim is not None
                and ultim["tip"] == "desen" and ultim["id"] == intrare["id"]):
            self.istoric[-1] = intrare
        else:
            self.istoric.append(intrare)
            del self.istoric[:-MAX_ISTORIC]
        self._deseneaza_harta()

    def sterge_istoric(self):
        self.istoric = []
        self._deseneaza_harta()

    def _deseneaza_harta(self):
        ax, h, c = self.ax_harta, self.harta, CULOARE[self.activ]
        ax.clear()
        ax.set_facecolor(FUNDAL)
        P = h["P"]
        ax.scatter(P[:, 0], P[:, 1], s=2, color="#55544f", alpha=0.6, linewidths=0)
        for cif, (px, py) in h["etichete"].items():
            ax.text(px, py, str(cif), fontsize=10, fontweight="bold", ha="center", va="center",
                    color="#ffffff", bbox=dict(boxstyle="circle,pad=0.2", fc="#2a2a28",
                                               ec="#6b6a64", alpha=0.85))

        vizibile = [e for e in self.istoric if e["tip"] == "desen" or e["model"] == self.activ]
        desene = [e for e in vizibile if e["tip"] == "desen"]
        if desene:
            lat, sig = self._latent(np.stack([e["x"] for e in desene]))
            for e, l, s in zip(desene, lat, sig if sig is not None else [None] * len(desene)):
                e["_p"], e["_sigma"] = h["pca"].aplica(l)[0], s
        for e in vizibile:
            if e["tip"] == "generat":
                e["_p"] = h["pca"].aplica(e["z"])[0]

        for k, e in enumerate(vizibile):
            ultim = k == len(vizibile) - 1
            alfa = 1.0 if ultim else 0.45
            px, py = e["_p"]
            if e["tip"] == "desen" and e.get("_sigma") is not None:
                w, hh, ang = h["pca"].elipsa(e["_sigma"], k=2)
                ax.add_patch(Ellipse((px, py), w, hh, angle=ang, fc=c, ec=c,
                                     alpha=0.28 if ultim else 0.1, lw=1, zorder=2))
            marker = "X" if e["tip"] == "generat" else "o"
            ax.scatter([px], [py], s=110 if ultim else 45, marker=marker, color=c,
                       edgecolors=FUNDAL, linewidths=1.5, alpha=alfa, zorder=4)
            ax.annotate(str(k + 1), (px, py), xytext=(6, 5), textcoords="offset points",
                        fontsize=8, color="#ffffff" if ultim else "#c3c2b7", zorder=5)

        ax.set_title("{} ({}): spatiu latent, PCA 2D, {:.0f}% din variatie".format(
            self.activ, self.epoca_sel.get(), h["pca"].varianta_pastrata * 100),
            fontsize=9, color="#ffffff")
        ax.tick_params(colors="#8a8984", labelsize=7)
        for s in ax.spines.values():
            s.set_color("#3a3a37")
        ax.text(0.01, 0.01, "o desen" + ("  (nor 2σ)" if self.activ == "VAE" else "")
                + "    X generat din N(0,1)", transform=ax.transAxes,
                fontsize=7, color="#c3c2b7", va="bottom")
        self.canvas_harta.draw_idle()

    # ------------------------------------------------------------------
    def _construieste_interfata(self):
        # --- deasupra tuturor: filtrele invatate de primul strat ---
        cap = tk.Frame(self.radacina, bg=FUNDAL)
        cap.pack(padx=10, pady=(10, 4), fill=tk.X)
        tk.Label(cap, text="CE CAUTA primul strat: greutatile a 32 din cei 128 "
                           "de neuroni, desenate ca imagini 28x28",
                 bg=FUNDAL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack()
        tk.Label(cap, text="rosu = greutate pozitiva (cauta cerneala aici)   |   "
                           "albastru = negativa (cauta absenta cernelii)",
                 bg=FUNDAL, fg="#888", font=("Consolas", 8)).pack()

        rand_model = tk.Frame(cap, bg=FUNDAL)
        rand_model.pack(pady=(4, 0))
        tk.Label(rand_model, text="model:", bg=FUNDAL, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=(0, 6))
        self.model_sel = tk.StringVar(value=self.activ)
        for nume in self.modele:
            tk.Radiobutton(rand_model, text=nume, variable=self.model_sel, value=nume,
                           command=self.schimba_model,
                           bg=FUNDAL, fg=TEXT, selectcolor="#333",
                           activebackground=FUNDAL, activeforeground=TEXT,
                           font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=4)

        self.rand_epoci = tk.Frame(cap, bg=FUNDAL)
        self.rand_epoci.pack(pady=3)
        self.epoca_sel = tk.StringVar()
        self.et_info_ep = tk.Label(cap, text="", bg=FUNDAL, fg="#7aa37a",
                                   font=("Consolas", 8))
        self.et_info_ep.pack()
        self._reface_selector_epoci()

        self.et_filtre = tk.Label(cap, bg="black")
        self.et_filtre.pack(pady=3)

        sus = tk.Frame(self.radacina, bg=FUNDAL)
        sus.pack(padx=10, pady=(4, 10))

        # --- stanga: zona de desen ---
        stanga = tk.Frame(sus, bg=FUNDAL)
        stanga.pack(side=tk.LEFT, padx=(0, 14))
        tk.Label(stanga, text="Deseneaza o cifra", bg=FUNDAL, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack()
        self.panza = tk.Canvas(stanga, width=PANZA, height=PANZA,
                               bg="black", highlightthickness=1,
                               highlightbackground="#555")
        self.panza.pack(pady=4)
        self.panza.bind("<B1-Motion>", self._deseneaza)
        self.panza.bind("<ButtonRelease-1>", self._ridica)

        butoane = tk.Frame(stanga, bg=FUNDAL)
        butoane.pack(pady=4)
        tk.Button(butoane, text="Ruleaza", command=self.ruleaza,
                  width=11, bg="#2d6a4f", fg="white").pack(side=tk.LEFT, padx=3)
        tk.Button(butoane, text="Sterge", command=self.sterge,
                  width=11, bg="#6a2d2d", fg="white").pack(side=tk.LEFT, padx=3)
        tk.Button(butoane, text="Genereaza", command=self.genereaza,
                  width=11, bg="#2d4a6a", fg="white").pack(side=tk.LEFT, padx=3)

        self.preproc = tk.BooleanVar(value=True)
        tk.Checkbutton(stanga, text="preprocesare MNIST (centrare pe masa)",
                       variable=self.preproc, command=self._comuta_preproc,
                       bg=FUNDAL, fg=TEXT, selectcolor="#333",
                       activebackground=FUNDAL, activeforeground=TEXT).pack()

        self.stare = tk.Label(stanga, text="deseneaza si apasa Ruleaza",
                              bg=FUNDAL, fg="#999", font=("Consolas", 8),
                              justify=tk.LEFT)
        self.stare.pack(pady=(6, 0))

        # --- dreapta: intrare procesata si reconstructie ---
        dreapta = tk.Frame(sus, bg=FUNDAL)
        dreapta.pack(side=tk.LEFT)
        self.et_intrare = self._casuta(dreapta, "intrare 28x28 (ce vede modelul)")
        self.et_iesire = self._casuta(dreapta, "reconstructie (ce a produs)")
        self.et_eroare = self._casuta(dreapta, "eroare (unde a gresit)")

        # --- dreapta de tot: harta spatiului latent, cu istoricul desenelor ---
        cadru_harta = tk.Frame(sus, bg=FUNDAL)
        cadru_harta.pack(side=tk.LEFT, padx=(14, 0))
        fig = Figure(figsize=(4.8, 4.4), dpi=90, facecolor=FUNDAL)
        self.ax_harta = fig.add_axes([0.08, 0.07, 0.9, 0.86])
        self.canvas_harta = FigureCanvasTkAgg(fig, master=cadru_harta)
        self.canvas_harta.get_tk_widget().pack()
        tk.Button(cadru_harta, text="Sterge istoricul hartii", command=self.sterge_istoric,
                  bg="#3a3a37", fg="white").pack(pady=(4, 0))

        # --- jos: activarile pe straturi ---
        jos = tk.Frame(self.radacina, bg=FUNDAL)
        jos.pack(padx=10, pady=(0, 10))
        tk.Label(jos, text="ENCODER  (comprima)          z          DECODER  (reface)",
                 bg=FUNDAL, fg=TEXT, font=("Segoe UI", 10, "bold")).pack()
        randul = tk.Frame(jos, bg=FUNDAL)
        randul.pack()
        self.panouri = {}
        for nume, eticheta in [("enc_1", "128"), ("enc_2", "64"), ("z", "32 = z"),
                               ("dec_1", "64"), ("dec_2", "128")]:
            cadru = tk.Frame(randul, bg=FUNDAL)
            cadru.pack(side=tk.LEFT, padx=5)
            lbl = tk.Label(cadru, bg="black", width=96, height=96)
            lbl.pack()
            tk.Label(cadru, text=eticheta, bg=FUNDAL, fg=TEXT,
                     font=("Consolas", 9)).pack()
            info = tk.Label(cadru, text="", bg=FUNDAL, fg="#888",
                            font=("Consolas", 7))
            info.pack()
            self.panouri[nume] = (lbl, info)

    def _deseneaza_filtre(self, n_filtre=32, pe_rand=16, marime=52):
        """
        Deseneaza greutatile primului strat ca imagini 28x28.

        Matricea stratului enc_1 are forma (784, 128): coloana i contine
        greutatile neuronului i, cate una pentru fiecare pixel de intrare.
        Le asezam inapoi in 28x28 si obtinem literalmente tiparul pe care
        acel neuron il cauta in imagine.

        Normalizam fiecare filtru separat, pe maximul lui absolut: valorile
        brute difera mult de la un neuron la altul, iar fara normalizare
        individuala majoritatea ar aparea negre.
        """
        W = self._m()["encoder"].get_layer("enc_1").get_weights()[0]  # (784, 128)

        # Alegem neuronii cu cea mai mare amplitudine: cei apropiati de zero
        # nu au invatat nimic vizibil si ar umple grila cu patrate goale.
        putere = np.abs(W).sum(axis=0)
        alesi = np.argsort(putere)[::-1][:n_filtre]

        randuri = int(np.ceil(n_filtre / pe_rand))
        panza = Image.new("RGB", (pe_rand * (marime + 2), randuri * (marime + 2)),
                          (20, 20, 20))
        for k, idx in enumerate(alesi):
            f = W[:, idx].reshape(28, 28)
            vmax = np.abs(f).max() or 1.0
            rgb = np.zeros((28, 28, 3), dtype="uint8")
            rgb[..., 0] = (np.clip(f / vmax, 0, 1) * 255).astype("uint8")
            rgb[..., 2] = (np.clip(-f / vmax, 0, 1) * 255).astype("uint8")
            mic = Image.fromarray(rgb, "RGB").resize((marime, marime), Image.NEAREST)
            panza.paste(mic, ((k % pe_rand) * (marime + 2),
                              (k // pe_rand) * (marime + 2)))

        foto = ImageTk.PhotoImage(panza)
        self.et_filtre.configure(image=foto)
        self.et_filtre.image = foto

    def _casuta(self, parinte, titlu):
        cadru = tk.Frame(parinte, bg=FUNDAL)
        cadru.pack(pady=3)
        tk.Label(cadru, text=titlu, bg=FUNDAL, fg=TEXT,
                 font=("Segoe UI", 8)).pack()
        lbl = tk.Label(cadru, bg="black", width=112, height=112)
        lbl.pack()
        return lbl

    # ------------------------------------------------------------------
    def _deseneaza(self, ev):
        if self.ultim is not None:
            self.panza.create_line(self.ultim[0], self.ultim[1], ev.x, ev.y,
                                   fill="white", width=GROSIME,
                                   capstyle=tk.ROUND, smooth=True)
            self.creion.line([self.ultim, (ev.x, ev.y)], fill=255,
                             width=GROSIME, joint="curve")
        self.ultim = (ev.x, ev.y)

    def _ridica(self, _):
        self.ultim = None
        self.ruleaza()

    def sterge(self):
        self.panza.delete("all")
        self.desen = Image.new("L", (PANZA, PANZA), 0)
        self.creion = ImageDraw.Draw(self.desen)
        self.stare.config(text="deseneaza si apasa Ruleaza")
        self.id_desen += 1          # urmatorul desen primeste punct nou pe harta

    def _comuta_preproc(self):
        # Acelasi desen, alta preprocesare: punct nou, ca sa se vada pe harta
        # cat de departe muta centrarea intrarea in spatiul latent.
        self.id_desen += 1
        self.ruleaza()

    # ------------------------------------------------------------------
    def ruleaza(self, adauga=True):
        arr = np.asarray(self.desen, dtype="uint8")
        if arr.max() < 20:
            return

        yy, xx = np.mgrid[0:28, 0:28]
        if self.preproc.get():
            img, pasi = desen_ca_mnist(arr)
            cy, cx = pasi.get("centru_masa", (0.0, 0.0))
            detaliu = "centru masa {:.1f},{:.1f} -> 13.5,13.5".format(cy, cx)
        else:
            img = desen_brut(arr)
            t = img.sum() or 1
            detaliu = "centru masa {:.1f},{:.1f} (necentrat)".format(
                (img * yy).sum() / t, (img * xx).sum() / t)

        x = img.reshape(1, 784).astype("float32")
        a1, a2, z = [np.asarray(v)[0] for v in self._m()["m_enc"].predict(x, verbose=0)]
        d1, d2, rec = [np.asarray(v)[0] for v in self._m()["m_dec"].predict(
            z.reshape(1, -1), verbose=0)]

        self._pune_imagine(self.et_intrare, img)
        self._pune_imagine(self.et_iesire, rec.reshape(28, 28))
        self._pune_imagine(self.et_eroare, np.abs(img - rec.reshape(28, 28)), harta="hot")

        for nume, vect in [("enc_1", a1), ("enc_2", a2), ("z", z),
                           ("dec_1", d1), ("dec_2", d2)]:
            self._pune_activari(nume, vect)

        eroare = float(np.abs(img - rec.reshape(28, 28)).mean())
        zerouri = int((np.abs(z) < 1e-6).sum())
        self.stare.config(
            text="{}: eroare medie/pixel: {:.4f}\n{}\nz: min {:+.2f}  max {:+.2f}  zerouri {}/{}".format(
                self.activ, eroare, detaliu, z.min(), z.max(), zerouri, len(z)))
        if adauga:
            self._adauga_istoric({"tip": "desen", "x": x[0], "id": self.id_desen})

    # ------------------------------------------------------------------
    def _pune_imagine(self, eticheta, arr2d, harta=None):
        a = np.clip(arr2d, 0, 1)
        if harta == "hot":
            # rosu-galben pentru eroare, ca sa se distinga de gri
            rgb = np.zeros((28, 28, 3), dtype="uint8")
            rgb[..., 0] = (np.clip(a * 3, 0, 1) * 255).astype("uint8")
            rgb[..., 1] = (np.clip(a * 3 - 1, 0, 1) * 255).astype("uint8")
            img = Image.fromarray(rgb, "RGB")
        else:
            img = Image.fromarray((a * 255).astype("uint8"), "L")
        img = img.resize((112, 112), Image.NEAREST)
        foto = ImageTk.PhotoImage(img)
        eticheta.configure(image=foto, width=112, height=112)
        eticheta.image = foto

    def _pune_activari(self, nume, vect):
        """
        Deseneaza un vector de activari ca grila. Normalizam pe maximul
        vectorului curent: ne intereseaza care neuroni sunt aprinsi relativ
        unii la altii, nu valoarea absoluta, care difera mult intre straturi.
        """
        eticheta, info = self.panouri[nume]
        n = len(vect)
        lat = int(np.ceil(np.sqrt(n)))
        inalt = int(np.ceil(n / lat))
        grila = np.zeros(lat * inalt, dtype="float32")
        grila[:n] = vect
        grila = grila.reshape(inalt, lat)

        vmax = np.abs(grila).max() or 1.0
        if vect.min() < 0:
            # z poate fi negativ: albastru pentru negativ, rosu pentru pozitiv
            rgb = np.zeros((inalt, lat, 3), dtype="uint8")
            rgb[..., 0] = (np.clip(grila / vmax, 0, 1) * 255).astype("uint8")
            rgb[..., 2] = (np.clip(-grila / vmax, 0, 1) * 255).astype("uint8")
            img = Image.fromarray(rgb, "RGB")
        else:
            img = Image.fromarray((grila / vmax * 255).astype("uint8"), "L")

        img = img.resize((96, 96), Image.NEAREST)
        foto = ImageTk.PhotoImage(img)
        eticheta.configure(image=foto, width=96, height=96)
        eticheta.image = foto

        activi = int((np.abs(vect) > 1e-6).sum())
        info.config(text="{}/{} activi".format(activi, n))


if __name__ == "__main__":
    radacina = tk.Tk()
    Aplicatie(radacina)
    radacina.mainloop()
