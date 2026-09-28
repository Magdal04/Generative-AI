# Generative-AI
Posting Lab Work in the Course of Generative AI at my uni.

## Ce e aici

Fiecare folder e un laborator, si in fiecare raportul e notebook-ul `raportN_genai.ipynb`. Acolo e tot: explicatii, grafice, concluzii.

| folder | raport | ce e |
|---|---|---|
| `Lab1 GenAi` | `raport1_genai.ipynb` | generator de text caracter cu caracter (LSTM) |
| `Lab2 GenAi` | `raport2_genai.ipynb` | autoencoder vs VAE pe MNIST |
| `Lab3 GenAi` | `raport3_genai.ipynb` | GAN pe MNIST |

Modelele sunt deja antrenate si salvate, nu trebuie sa antrenezi nimic. Pe GitHub vezi rezultatele, dar partea interactiva merge doar daca rulezi local:

```
pip install tensorflow matplotlib pillow pandas ipywidgets jupyter
```

Deschizi notebook-ul din folderul lui (Jupyter sau VS Code) si dai **Run All**.

## Cum te joci cu ele

### Lab 1 - generator de text

**In notebook**, la sectiunea 6 ai casuta + slidere:
- scrii un inceput (`ARVINTE:`, `SCENA`, numele tau) si apesi **Genereaza**
- joaca-te cu **Temperature**: 0.2 intra in bucle, 0.8 e ok, 1.3 inventeaza cuvinte
- **Strategie** `argmax` = vezi cum se blocheaza repetand acelasi lucru

**In terminal**, tip chat:
```
cd "Lab1 GenAi"
python chat_live.py
```
Scrii un inceput, Enter, scrie litera cu litera. Comenzi: `/temp 1.2`, `/delay 0.3`, `/len 40`, `/mod sample`, `/quit`.

### Lab 2 - deseneaza o cifra

```
cd "Lab2 GenAi"
python app_desen.py
```
- desenezi cu mouse-ul, apesi **Ruleaza**: vezi reconstructia si ce se aprinde in fiecare strat
- **Genereaza** = cifra din z random, fara desen. Pe AE iese mazgaleala, pe VAE iese cifra
- sus schimbi **modelul** (AE / VAE) si **greutati de la** (epoca), ca sa vezi cum a invatat
- bifa **preprocesare MNIST**: acelasi desen cu si fara centrare, iese diferit
- ce desenezi apare pe harta latenta, **Sterge istoricul hartii** o curata

### Lab 3 - GAN

Nu are aplicatie, doar raportul. `REANTRENEAZA = False` foloseste ce e salvat, `True` reantreneaza tot (~15 min pe CPU).
