# Simulazione dei moti di deriva 

## Progetto dell'esame di Metodi Computazionali per la Fisica

Il programma simula i moti di deriva di particelle cariche con il drift $E\times B$ e $\nabla B\perp B$.
Si possono sia ricavare le traiettorie con il centro di guida per alcune particelle, sia studiare statisticamente come varia la velocità di deriva per diverse configurazioni di campo, verificando dunque la teoria dei moti di drift. 
Inoltre è possibile configurare un coefficiente di turbolenza che simula il comportamento delle turbolenze magnetiche che scatterano la direzione della particella in maniera casuale.

---
## Contenuto della repository

**Moti di deriva.pdf**:

File che descrive la teoria utilizzata per la simulazione, presenta anche una spiegazione di come è stata implementata e quali sono le condizioni parametriche da utilizzare. Inoltre discute anche dell'analisi dei dati che viene svolta e dei risultati ricavati.

**Bibliografia**:

Cartella con i file .pdf utilizzati per la teoria alla base della simulazione. Nello specifico si trovano i seguenti testi:

   * Longair, High Energy Astrophysics, third edition

   * Gould, An Introduction to Computer Simulation Methods, first edition

   * Birdsall & Langdon, Plasma Physics via computer simulation
   
   * Tomassetti, Astrofisica dei Raggi Cosmici: Fenomenologia e Modelli

   * Tomassetti, Fisica dello spazio: Teoria delle orbite
   
**Simulazione**: 

Cartella contenti i codici e dati utilizzati per la simulazione, nello specifico si ha:

   * drift_data.csv: file che contiene i dati salvati della simulazione
	
   * main.py: script principale da eseguire per avviare la simulazione
	
   * drift_motions.py: script che implementa le funzioni che ricavano la traiettoria della particella, il centro di guida e la veloità di drift. Compare anche la funzione che randomizza la direzione della particella in caso di turbolenza.
   
   * analysis.py: script che implementa le funzioni utilizzate per l'indagine statistica delle velocità di deriva. Calcola sia la velocità media di deriva per una simulazione di $1000$ particelle, sia genera il fit per diverse simulazioni verificando la linearità della velocità per diverse configurazioni di campi (Utilizza i dati salvati in drift_data.csv).
   
   * plots.py: script che implementa i codici utilizzati per produrre i grafici delle traiettorie delle particelle, le distribuzioni delle componenti della velocità per una simulazione di $1000$ particelle e il grafico del fit lineare per diverse velocità di deriva medie.
   
---
## Librerie utilizzate

* Os
* Sys
* Numpy
* Scipy
* Pandas
* Argparse
* Matplotlib
* Tqdm

---
## Installazione

Per ottenere il codice della simulazione, digitare nel terminale la seguente riga di comando:

'''bash
git clone https://github.com/DavidNuccioni/PROGETTO_MCF.git

---
## Obiettivo e fisica della simulazione

Il programma vuole implementare il comportamento delle particella poste in campi magnetici, dove sono presenti o dei campi elettrici o dei gradienti magnetici perpendicolari al campo magentico. 
La presenza di questi ultimi genera una velocità di deriva per la particella, descritta da:

$$v_d = \frac{E\times B}{B^2}$$

Nel caso del campo elettrico, oppure:

$$v_d = \pm\frac{1}{2}v_{perp}r_L\frac{\nabla B\times B}{b^2}$$

Nel caso del gradiente magnetico. 
La velocità di deriva devia il moto della particella rispetto al semplice moto ciclotronico che si avrebbe nel caso fosse presente solo il campo magnetico della particella. Ci si aspetta che la simulazione riesca dunque a riprodurre i risultati teorici. Per farlo vengono simulate miglaia di particelle con diverse configurazioni delle intensità di $E$ e di $\nabla B$, ci si aspetta che la velocità di deriva cresca linearmente con queste diverse configurazioni, considerando delle particelle cariche positivamente e negativamente e velocità iniziali casuali. Per maggiori informazioni riguardo la teoria dei moti di deriva consultare il capitolo 1 del file: Moti di deriva.pdf.

Per ricavare la traiettoria della particella nei due casi di drift, viene implementato l'algortimo di Boris che permette di descrivere come si muove una particella posta in campi elettrici e magnetici. Tramite poi il moto ciclotronico della particella si ricava la traiettoria del centro di guida, tramite una funzione che media su ogni orbita della particella e ne ricava il centro. Infine la velocità di deriva viene calcolata con una funzione che deriva semplicemente la velocità del centro di guida a partire dai punti ricavati precedentemente, a questo risultato viene tolto il contributo del moto ciclotronico ottenendo così la velcoità di deriva della particella simulata. Per maggiori informazioni sull'implementazione dell'algortimo della simulazione, consultare il capitolo 2 del file: Moti di deriva.pdf.

La capacità della simulazione, nel riprodurre dei risultati veritieri, è stata studiata tramite indagine statistica sui due diversi tipi di drift. Prima si ricavava il modulo della velocità media di deriva tramite simulazione per un campione di $1000$ particelle, verificando dunque l'accordo con la previsione teorica. Successivamente si è eseguito un fit lineare sulle velocità di deriva medie, con diverse configurazioni di campi elettrici o di gradienti magnetici, verificando la dipendenza lineare della velocità da questi. I risultati ricavati sono in accordo con la teoria. Per maggiori informazioni sull'analisi dati della simulazione consultare il capitolo 3 del file: Moti di deriva.pdf.

---
## Struttura del programma




---
## Argomenti del programma




---
## Utilizzo della simulazioni




---
## Configurazione dei campi e range dei parametri




---
## Esempi di utilizzo 



















   
   
   
   
   
	
---			
