


# Autoencoder clasic — flux de date

Imaginea intră ca 784 de pixeli, e strânsă printr-un gât îngust de 32 de numere,
apoi reconstruită tot la 784. Nu există etichete: ținta loss-ului este chiar
imaginea de intrare.

```mermaid
flowchart LR
    X["Imagine 28x28<br/>784 pixeli"] --> E["ENCODER<br/>784 - 128 - 64"]
    E --> Z["z<br/>32 numere"]
    Z --> D["DECODER<br/>64 - 128 - 784"]
    D --> R["Reconstructie<br/>784 pixeli"]
    X -.->|"MSE loss"| R
    style Z fill:#ffd966,stroke:#333,stroke-width:2px
```

Săgeata punctată este esența: modelul se compară cu propria intrare.
Dacă gâtul `z` ar avea 784 de  
dimensiuni, rețeaua ar putea copia intrarea
pixel cu pixel și nu ar învăța nimic despre structura datelor.

# AE clasic vs. VAE — ce se schimbă în gât

Autoencoderul clasic trimite spre decoder un singur punct. VAE trimite
parametrii unei distribuții și eșantionează din ea, ceea ce face spațiul
latent continuu și deci utilizabil pentru generare.

```mermaid
flowchart LR
    subgraph AE["AE clasic"]
        direction LR
        E1["encoder"] --> Z1["z: 32 numere<br/>UN PUNCT"]
        Z1 --> D1["decoder"]
    end
    subgraph VAE["VAE"]
        direction LR
        E2["encoder"] --> M["mu: 32"]
        E2 --> V["log_var: 32"]
        M --> S["z = mu + sigma * eps<br/>UN NOR"]
        V --> S
        S --> D2["decoder"]
    end
    style Z1 fill:#ffd966,stroke:#333,stroke-width:2px
    style S fill:#9fd6a0,stroke:#333,stroke-width:2px
```

La VAE, `eps` este zgomot aleator standard-normal. Reparametrizarea
`z = mu + sigma * eps` mută aleatorul în afara drumului gradientului,
astfel încât backpropagation să poată în continuare ajusta `mu` și `log_var`.



# De ce AE clasic nu poate genera

Ambele modele reconstruiesc bine. Diferența apare când eșantionezi un `z`
aleator și îl dai direct decoderului, fără să fi trecut printr-o imagine reală.

```mermaid
flowchart TB
    subgraph AEL["Spatiu latent AE clasic"]
        A1["insula cifre 1"]
        A2["insula cifre 7"]
        A3["insula cifre 0"]
        AG["gauri mari nelocuite"]
    end
    subgraph VAEL["Spatiu latent VAE"]
        B1["nor compact continuu<br/>centrat in origine"]
    end
    RA["z aleator"] --> AG
    AG --> BAD["decoder produce zgomot"]
    RB["z aleator"] --> B1
    B1 --> GOOD["decoder produce cifra plauzibila"]
    style BAD fill:#f4a6a6,stroke:#333
    style GOOD fill:#9fd6a0,stroke:#333
```

Termenul KL din loss-ul VAE este forța care strânge toate codurile spre
origine. Fără el, VAE degenerează într-un autoencoder obișnuit.

```mermaid
784 pixeli  --ReLU-->  128         --ReLU-->  64          --ReLU-->  32
imagine bruta          segmente,              bucle,                 "ce fel
                       capete de linie        intersectii,           de cifra,
                                              colturi                cum scrisa"

            trasaturi              trasaturi               esenta
            simple                 compuse

```
```mermaid
ENCODER:  784 -> 128 -> 64 -> 32
                                  z
DECODER:                     32 -> 64 -> 128 -> 784

```

```mermaid
flowchart LR
    X["784<br/>pixeli"] --> H1["128<br/>ReLU"]
    H1 --> H2["64<br/>ReLU"]
    H2 --> Z["32<br/>liniar"]
    Z --> H3["64<br/>ReLU"]
    H3 --> H4["128<br/>ReLU"]
    H4 --> Y["784<br/>sigmoid"]
    style Z fill:#ffd966,stroke:#333,stroke-width:2px
    style Y fill:#9fd6a0,stroke:#333
```
