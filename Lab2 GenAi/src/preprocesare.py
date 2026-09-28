"""
Aducerea unui desen facut de utilizator in forma datelor MNIST.

MNIST nu contine imagini brute de 28x28. Procedura originala (Yann LeCun):
cifra e decupata la bounding box, scalata astfel incat latura mare sa fie
20 de pixeli, apoi asezata intr-o caseta de 28x28 cu centrul de MASA in
mijloc, nu centrul geometric.

Fara acest pas, un desen ajunge la model cu alta pozitie, alta grosime si
alta scara decat tot ce a vazut la antrenare. Reconstructia iese proasta,
dar nu din vina modelului: intrarea e in afara distributiei pe care a
invatat-o. Modulul expune ambele variante ca sa se poata compara.
"""

import numpy as np
from PIL import Image


def desen_brut(arr):
    """
    Varianta naiva: doar redimensionare la 28x28 si normalizare in [0,1].
    Pastreaza pozitia si marimea din desen, care de obicei NU se potrivesc
    cu MNIST.

    arr: numpy 2D, valori 0..255, alb = cerneala.
    """
    img = Image.fromarray(arr.astype("uint8")).resize((28, 28), Image.LANCZOS)
    return np.asarray(img, dtype="float32") / 255.0


def desen_ca_mnist(arr):
    """
    Aplica procedura MNIST: decupare la bounding box, scalare la 20x20
    pastrand proportiile, centrare pe centrul de masa intr-o caseta 28x28.

    Returneaza (imagine_28x28_float, dictionar_cu_pasii) — pasii intra in
    interfata ca sa se vada ce s-a intamplat cu desenul.
    """
    a = arr.astype("float32")
    pasi = {}

    # 1. Bounding box: unde exista cerneala. Prag mic ca sa ignoram
    # eventuale urme de antialiasing aproape negre.
    masca = a > 20
    if not masca.any():
        return np.zeros((28, 28), dtype="float32"), {"gol": True}

    randuri = np.where(masca.any(axis=1))[0]
    coloane = np.where(masca.any(axis=0))[0]
    sus, jos = randuri[0], randuri[-1] + 1
    stanga, dreapta = coloane[0], coloane[-1] + 1
    decupat = a[sus:jos, stanga:dreapta]
    pasi["bbox"] = (int(sus), int(jos), int(stanga), int(dreapta))

    # 2. Scalare astfel incat latura MARE sa devina 20, proportiile pastrate.
    h, w = decupat.shape
    factor = 20.0 / max(h, w)
    h_nou, w_nou = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    mic = np.asarray(
        Image.fromarray(decupat.astype("uint8")).resize((w_nou, h_nou), Image.LANCZOS),
        dtype="float32",
    )
    pasi["dim_scalat"] = (h_nou, w_nou)

    # 3. Asezare provizorie in centrul geometric al unei casete 28x28.
    caseta = np.zeros((28, 28), dtype="float32")
    off_y, off_x = (28 - h_nou) // 2, (28 - w_nou) // 2
    caseta[off_y:off_y + h_nou, off_x:off_x + w_nou] = mic

    # 4. Deplasare astfel incat CENTRUL DE MASA sa cada in mijloc (13.5, 13.5).
    # Cerneala functioneaza ca greutate: un 7 are masa sus, un 6 are masa jos.
    # MNIST aliniaza dupa masa, nu dupa chenar, si de asta conteaza pasul.
    total = caseta.sum()
    yy, xx = np.mgrid[0:28, 0:28]
    cy, cx = (caseta * yy).sum() / total, (caseta * xx).sum() / total
    dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
    caseta = np.roll(np.roll(caseta, dy, axis=0), dx, axis=1)
    pasi["centru_masa"] = (float(cy), float(cx))
    pasi["deplasare"] = (dy, dx)

    return caseta / 255.0, pasi
