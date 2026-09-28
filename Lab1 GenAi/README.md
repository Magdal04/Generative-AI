# Generator de text caracter-cu-caracter cu LSTM

Un model care a citit ~1.4 milioane de caractere de literatură română și a învățat
singur să scrie text care *seamănă* cu ea — fără ca cineva să-i explice vreodată
ce e un cuvânt, o propoziție sau o literă.

---

## Ce face, pe scurt

Modelul primește un început de text și îl continuă, o literă pe rând.

```
Tu scrii:     ARVINTE: Ce
Modelul scrie: ARVINTE: Ce de dar mi-a spus...
```

Atât. Nu răspunde la întrebări, nu înțelege ce scrie. Prezice doar litera
următoare, iar dacă faci asta de 300 de ori la rând, iese text.

### Analogia care explică totul

Gândește-te la tastatura telefonului, care îți sugerează cuvântul următor.
Când scrii „mulțumesc frumos pentru...", telefonul ghicește ce urmează pentru că
a văzut milioane de mesaje și știe ce vine de obicei după.

Modelul ăsta face același lucru, cu două diferențe:
1. Ghicește **litere**, nu cuvinte
2. Îți ia sugestia automat și continuă, la nesfârșit

Dacă ai apăsa la infinit pe prima sugestie a telefonului, ai obține exact ce
face modelul ăsta.

---

## Rezultatele reale

Modelul a fost antrenat 47 de minute pe procesor (fără placă video). Uite cum a
evoluat, același început de text, la trei momente din antrenare:

**După 1 epocă** — știe că textul e făcut din „cuvinte" separate prin spații, dar
cuvintele nu există:
```
Untuncur, minărma cutări!
Nu puc în lo tis și locor fumeri suient
```

**După 11 epoci** — apar personaje, apare structura de dialog:
```
COȘCODAN: Un caselie se putrețură mîncă, stă Celor într-o săspunde
```

**După 20 de epoci** — cuvinte majoritar reale, gramatică aproape corectă:
```
PEPELEA: Iar pe urmele precum vine să primăserănască, înecare în față
în ceruri coastea cu ieși în sînge de mare giupâne!
```

Încă nu are sens. Dar are **formă** — și forma a fost învățată singură.

### Lucrul cel mai interesant

Nimeni nu i-a spus modelului ce e o piesă de teatru. Dar corpusul conținea
comedii de Alecsandri, iar modelul a dedus din statistică formatul:

```
SCENA VI
COȘCODAN, BARZĂ, zic: Ce-ai puș, îmi scoseau vesel închis
ARVINTE: Fețe, frace stînca cucrumul
```

A învățat că după `SCENA` vine o cifră romană, că un nume cu majuscule e urmat de
două puncte, și că apoi urmează replica. Din litere. Fără nicio regulă scrisă.

---

## Cum îl rulezi

### Să vezi text generat (rapid)
```bash
python charlstm.py generate
```
Scoate 4 texte, la 4 „temperaturi" diferite (vezi mai jos ce e aia).

### Să scrii tu începutul (interactiv)
```bash
python charlstm.py chat
```

### Să vezi literele apărând una câte una
```bash
python chat_live.py
```
Ca un chat, o literă pe secundă, cu procentele de încredere la final.

### Să vezi *cum gândește* la fiecare literă
```bash
python inspect_generation.py --n 15 --slow
```
Cel mai interesant. Arată, la fiecare pas, ce variante a luat în calcul.

### Să-l reantrenezi (47 de minute)
```bash
python charlstm.py train
```

---

## Conceptele, explicate simplu

### Caracter-cu-caracter (char-level)

Modelul lucrează cu litere individuale, nu cu cuvinte.

**De ce contează:** un model pe cuvinte are o listă fixă de cuvinte pe care le
știe — să zicem 50.000. Dacă îi scrii un cuvânt care nu-i pe listă (un nume, o
poreclă, o greșeală de tipar), e complet blocat. Nu-l poate nici citi, nici scrie.
Se numește problema *out-of-vocabulary*.

Un model pe litere are doar 112 „cuvinte" de învățat — literele alfabetului,
cifre, semne de punctuație. Cu ele poate scrie **orice**, inclusiv cuvinte care
n-au existat niciodată.

**Analogie:** e diferența dintre a memora un dicționar (limitat la ce scrie în el)
și a învăța alfabetul (poți scrie orice, inclusiv „supercalifragilistic").

### Hidden state (starea ascunsă)

Problema: dacă modelul vede doar litera curentă, n-are cum să ghicească nimic.
După litera `a` poate veni orice.

Soluția: modelul ține o „memorie de lucru" — un set de 256 de numere care se
actualizează după fiecare literă și rezumă tot ce-a citit până atunci.

**Analogie:** citești o carte și cineva te întreabă ce urmează. Nu te uiți doar la
ultimul cuvânt — ai în cap tot ce s-a întâmplat: cine sunt personajele, unde sunt,
ce tocmai a zis cineva. Dar nu ții minte fiecare cuvânt din carte, ci un rezumat.
Hidden state e rezumatul ăla.

Important: nu crește. Rămâne mereu 256 de numere, fie că ai citit 10 litere sau
10.000. Ce nu încape, se pierde — de-asta modelul uită începutul propoziției.

### LSTM

Un tip de rețea care decide singură ce să țină minte și ce să uite din rezumat.
Are niște „porți" interne care controlează ce informație trece mai departe.

**De ce a fost inventat:** rețelele recurente simple, mai vechi, uitau aproape
instant. După 10-15 litere, informația se dizolva. LSTM ține context pe sute de
pași.

**Analogie:** iei notițe la curs. Nu scrii tot ce zice profesorul — decizi
continuu ce e important de notat, ce poți lăsa, și când să tai ceva ce ai notat
mai devreme și nu mai contează. Porțile LSTM fac exact deciziile astea, automat.

### Temperature (temperatura)

Cel mai distractiv buton. Controlează cât de riscant alege modelul.

La fiecare literă, modelul nu alege una singură — calculează șanse pentru toate
cele 112. De exemplu după `Ce de`:

```
spațiu  79.4%  ███████████████████████
   -    14.3%  ████
   s     1.7%
   ș     1.0%
```

Temperatura decide cât de mult contează diferențele astea:

| temperatură | ce face | rezultat |
|---|---|---|
| 0.2 | alege aproape mereu varianta cea mai probabilă | corect, dar se repetă la nesfârșit |
| 0.8 | echilibru | cuvinte reale, variație bună |
| 1.5 | dă șanse mari și variantelor improbabile | inventează cuvinte, haotic |

**Analogie:** e ca la karaoke. Temperatura mică = cânți exact melodia, sigur, dar
plictisitor. Temperatura mare = improvizezi, uneori iese genial, de obicei iese
fals. 0.8 e unde e distracția fără dezastru.

Exemple reale din model:

*Temperatura 0.2* (se blochează în buclă):
```
în care mi-a pus în care mi-a pus în care mi-a pus
```

*Temperatura 1.2* (inventează cuvinte):
```
Merteva protrec… Am... ești? „Pe-un cale adăug margănăfopotenită
```

### Sampling (eșantionare)

Modelul **nu** alege întotdeauna varianta cea mai probabilă. Trage la sorți,
proporțional cu șansele.

Dacă o literă are 5%, e aleasă în 5% din cazuri. Rar — dar se întâmplă.

**De ce nu alegem mereu maximul:** pentru că atunci modelul devine previzibil și
intră în bucle. Ai văzut mai sus ce iese la temperatură mică — exact asta.

**Analogie:** Spotify shuffle. Dacă ar pune mereu melodia ta preferată, ai asculta
aceeași piesă la infinit. Faptul că uneori pune ceva neașteptat e ce face
playlistul interesant.

### Epocă

O rundă de antrenare. Modelul citește o parte din text, își corectează greșelile,
o ia de la capăt.

Aici, o „epocă" = 250 de runde × 256 de exemple = 64.000 de exemple, adică 23% din
date. După 20 de epoci, modelul a trecut de ~4.6 ori prin tot corpusul.

**Analogie:** înveți la un examen. Prima citire pricepi ideea generală. A doua
prinzi detalii. A patra, a cincea — începi să nu mai câștigi mare lucru. Exact ce
s-a întâmplat și aici (vezi tabelul de mai jos).

---

## Numerele proiectului

### Model
| | |
|---|---|
| Parametri | 364.656 |
| Mărime pe disc | 1.39 MB |
| Vocabular | 112 caractere unice |
| Corpus | 1.385.564 caractere |

Observație: corpusul are 1.38 milioane de caractere, modelul ocupă 1.39 MB.
Textul **nu încape** în model — deci modelul nu-l memorează, îl rezumă.

### Antrenare
| | |
|---|---|
| Durată | 47.3 minute |
| Hardware | procesor, fără placă video |
| Loss inițial | 2.482 |
| Loss final | 1.549 |

**Ce e loss-ul:** cât de greșit prezice modelul. Mai mic = mai bun. La 1.549
înseamnă că modelul ezită efectiv între ~5 litere la fiecare pas, în loc de 112
dacă ar ghici la întâmplare.

Câștigul pe epoci:
```
epoca  1: 2.48  ██████████████████████████
epoca  5: 1.82  ████████████████
epoca 10: 1.62  ████████████
epoca 15: 1.57  ███████████
epoca 20: 1.55  ██████████
```

Se vede clar: după epoca 10, îmbunătățirea aproape se oprește. Ultimele 10 epoci
au adus 0.07 — de-asta antrenarea s-a oprit la 20.

### Viteza, măsurată pe acest procesor

Toate numerele astea vin din rulare reală ([probe.py](probe.py)), nu din estimări:

| unități | batch | lungime secvență | secunde/pas | caractere/secundă |
|---|---|---|---|---|
| 128 | 128 | 100 | 0.153 | 83.647 |
| **256** | **256** | **100** | **0.716** | **35.730** ← ales |
| 256 | 128 | 100 | 0.468 | 27.322 |
| 256 | 128 | 50 | 0.305 | 20.978 |
| 512 | 128 | 100 | 0.904 | 14.166 |

Ce se învață din tabel:
- Dublarea de la 256 la 512 unități costă **3.1x** mai mult timp
- Batch mai mare = mai lent per pas, dar **mai rapid per caracter** (31% câștig)
- Secvențe mai scurte (50) sunt **mai proaste** — costul fix per pas se împarte la
  mai puține caractere

---

## Verificări făcute

### Memorează sau compune?

Cea mai lungă secvență generată care apare cuvânt-cu-cuvânt în corpus: **24 de
caractere** din 1000 generate. Scurt — modelul compune, nu copiază.

Dovadă suplimentară: cuvinte care nu există în română și deci n-aveau cum să fie
în corpus — „protrec", „mnăvieți", „blâjăve", „cucrumul". Le-a inventat din
statistica literelor.

### A învățat ortografia românească?

Frecvența diacriticelor, generat vs. original:

| literă | în text generat | în corpus | raport |
|---|---|---|---|
| ă | 2.78% | 3.34% | 0.83 |
| â | 0.80% | 0.88% | 0.90 |
| î | 0.90% | 0.87% | **1.03** |
| ș | 0.80% | 1.39% | 0.57 |
| ț | 0.80% | 0.74% | **1.07** |

Raport 1.00 = frecvență identică cu originalul. `î` și `ț` sunt aproape perfecte.
`ș` e subutilizat — apare des în terminații (`ești`, `așa`) care cer context mai
lung decât reține modelul bine.

---

## Ce NU poate face

Important de înțeles, pentru că e ușor de confundat cu ChatGPT:

- **Nu răspunde la întrebări.** Dacă scrii „Ce mai faci?", continuă textul, nu
  răspunde. Nu știe ce e o întrebare.
- **Nu înțelege sensul.** Zero. Știe doar ce literă vine statistic după alta.
- **Uită repede.** Ține minte ~100 de caractere, cam 15 cuvinte.
- **Nu poate învăța lucruri noi** fără reantrenare completă.

**Comparația care pune totul în context:** modelul ăsta are 364.656 de parametri.
GPT-2 „small", lansat în 2019 și considerat azi minuscul, are 124.000.000 — de
340 de ori mai mult. Modelele actuale au sute de miliarde.

E un instrument de studiu, nu o unealtă. Dar mecanismul de bază — prezici ce
urmează, eșantionezi, repeți — e **exact același** ca la modelele mari. Doar
scara diferă.

---

## Fișierele proiectului

| fișier | ce face |
|---|---|
| [charlstm.py](charlstm.py) | inima proiectului: date, model, antrenare, generare |
| [probe.py](probe.py) | măsoară viteza pe 5 configurații, înainte de antrenare |
| [evaluate.py](evaluate.py) | verifică memorarea, diacriticele, progresia |
| [inspect_generation.py](inspect_generation.py) | arată deciziile pas cu pas |
| [chat_live.py](chat_live.py) | chat cu literele apărând una câte una |
| `checkpoints/` | modelul salvat după fiecare epocă |
| `checkpoints/history.csv` | loss-ul la fiecare epocă |

### Butoanele pe care le poți schimba

În [charlstm.py](charlstm.py), liniile 41-52:

```python
SEQ_LEN = 100        # câte litere vede înainte să prezică
STRIDE = 5           # cât de mult se suprapun exemplele de antrenare
BATCH_SIZE = 256     # câte exemple procesează odată
EMBED_DIM = 64       # câte numere descriu fiecare literă
LSTM_UNITS = 256     # mărimea memoriei de lucru
EPOCHS = 20          # câte runde de antrenare
LEARNING_RATE = 2e-3 # cât de mari sunt corecțiile
```

Singurul care nu cere reantrenare: **temperature**, la generare. Schimbă complet
rezultatul, costă zero.

---

## Detalii tehnice rezolvate pe parcurs

Lucruri care par mărunte dar rup proiectul:

**Normalizare Unicode (NFC).** Litera `ș` poate fi salvată în două feluri diferite
în calculator: ca un singur caracter, sau ca `s` + sedila lipită de el. Arată
identic pe ecran, dar pentru model sunt două litere complet diferite. Fără
normalizare, vocabularul s-ar umple cu duplicate. Se rezolvă într-o linie, dar
dacă o ratezi, strici statistica fără să observi.

**Consola Windows.** Terminalul Windows folosește implicit o codare veche
(cp1252) care nu știe să afișeze `Ț`. Programul crăpa la prima literă românească
afișată — nu la model, la `print`. Rezolvat cu un wrapper UTF-8 pe ieșire.

**Testarea generării înainte de antrenare.** Codul de generare se testează pe
modelul **neantrenat**, înainte să înceapă antrenarea. Textul iese gunoi — e în
regulă, nici nu contează. Ce contează e că un bug în generare se vede în primul
minut, nu după 47 de minute de antrenare.

**Checkpoint după fiecare epocă.** Modelul se salvează la fiecare ~2 minute. Dacă
laptopul se oprește la minutul 40, nu pierzi nimic. În plus, checkpoint-urile sunt
ce permite comparația „epoca 1 vs epoca 11 vs epoca 20" de mai sus.
