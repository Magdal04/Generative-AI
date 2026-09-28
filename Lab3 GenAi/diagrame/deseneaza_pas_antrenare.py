"""Deseneaza diagrama celor doi pasi de antrenare GAN (diagrame/pas_antrenare.png)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

VERDE, GRI, ALBASTRU, PORTO, ROSU = "#2e7d32", "#9e9e9e", "#1565c0", "#ef6c00", "#c62828"

def cutie(ax, x, y, text, culoare, w=1.7, h=0.9):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.05",
                                fc=culoare, ec="black", lw=1.2))
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=11, weight="bold")

def sageata(ax, a, b, text="", culoare="black", stil="-|>", ls="-", rad=0.0, dy=0.25):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=stil, mutation_scale=18, color=culoare,
                                 lw=2, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + dy, text, ha="center", fontsize=10, color=culoare, bbox=dict(fc="white", ec="none", pad=1))

fig, axe = plt.subplots(2, 1, figsize=(12, 7.5))
for ax in axe:
    ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")

# ---- Pasul 1: D invata ----
ax = axe[0]
ax.set_title("Pasul 1 — Discriminatorul învață", fontsize=14, weight="bold", loc="left", color=VERDE)
cutie(ax, 1.2, 3.0, "Imagini\nREALE", ALBASTRU)
cutie(ax, 1.2, 1.0, "z aleator", PORTO)
cutie(ax, 3.8, 1.0, "G\n(doar produce)", GRI)
cutie(ax, 6.2, 1.0, "Imagini\nfalse", PORTO)
cutie(ax, 8.4, 2.0, "D\nÎNVAȚĂ", VERDE)
ax.text(10.9, 2.0, "reale → vrem 1\nfalse → vrem 0", ha="center", va="center", fontsize=11,
        bbox=dict(fc="white", ec="black"))
sageata(ax, (2.05, 3.0), (7.55, 2.2))
sageata(ax, (2.05, 1.0), (2.95, 1.0))
sageata(ax, (4.65, 1.0), (5.35, 1.0))
sageata(ax, (7.05, 1.0), (7.55, 1.8))
sageata(ax, (9.25, 2.0), (9.95, 2.0))
sageata(ax, (10.9, 1.45), (8.6, 1.5), "corectează D", ROSU, ls="--", rad=-0.5, dy=-1.05)

# ---- Pasul 2: G invata ----
ax = axe[1]
ax.set_title("Pasul 2 — Generatorul învață", fontsize=14, weight="bold", loc="left", color=VERDE)
cutie(ax, 1.2, 2.2, "z aleator\n(nou)", PORTO)
cutie(ax, 3.8, 2.2, "G\nÎNVAȚĂ", VERDE)
cutie(ax, 6.2, 2.2, "Imagini\nfalse", PORTO)
cutie(ax, 8.4, 2.2, "D\n(doar dă nota)", GRI)
ax.text(10.9, 2.2, "false → G vrea 1\n(„păcălește-l pe D”)", ha="center", va="center", fontsize=11,
        bbox=dict(fc="white", ec="black"))
for a, b in [((2.05, 2.2), (2.95, 2.2)), ((4.65, 2.2), (5.35, 2.2)),
             ((7.05, 2.2), (7.55, 2.2)), ((9.25, 2.2), (9.8, 2.2))]:
    sageata(ax, a, b)
sageata(ax, (10.9, 1.6), (3.9, 1.65), "corectează G — semnalul trece PRIN D, dar D nu se schimbă",
        ROSU, ls="--", rad=-0.25, dy=-1.45)
ax.text(1.2, 3.6, "Imaginile reale NU apar în acest pas", fontsize=10, style="italic", color=ALBASTRU)

fig.text(0.5, 0.01, "verde = rețeaua care își schimbă ponderile      gri = rețeaua doar consultată",
         ha="center", fontsize=10)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig("diagrame/pas_antrenare.png", dpi=110)
