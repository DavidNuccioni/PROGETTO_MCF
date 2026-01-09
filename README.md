# Simulazione dei moti di deriva 

## Progetto dell'esame di Metodi Computazionali per la Fisica

Il programma simula i moti di deriva di particelle cariche con il drift $E\times B$ e $\nabla B\perp B$.
Si possono sia ricavare le traiettorie con il centro di guida per alcune particelle, sia studiare statisticamente come varia la velocità di deriva per diverse configurazioni di campo, verificando dunque la teoria dei moti di drift. 
Inoltre è possibile configurare un coefficiente di turbolenza che simula il comportamento delle turbolenze magnetiche che scatterano la direzione della particella in maniera casuale.

---
# Contenuto della repository

**Moti_di_deriva.pdf**:

File che descrive la teoria utilizzata per la simulazione, presenta anche una spiegazione di come è stata implementata e quali sono le condizioni parametriche da utilizzare. Inoltre discute anche dell'analisi dei dati che viene svolta e dei risultati ricavati.

**Bibliografia**:

Cartella con i file .pdf utilizzati per la teoria alla base della simulazione. Nello specifico si trovano i seguenti testi:

   * Longair, High Energy Astrophysics, third edition

   * Gould, An Introduction to Computer Simulation Methods, first edition

   * Birdsall & Langdon, Plasma Physics via computer simulation, first edition
   
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
# Librerie utilizzate

* Os
* Sys
* Numpy
* Scipy
* Pandas
* Argparse
* Matplotlib
* Tqdm

---
# Installazione

Per ottenere il codice della simulazione, digitare nel terminale la seguente riga di comando:

```bash
git clone https://github.com/DavidNuccioni/PROGETTO_MCF.git
```

Il comando clonerà nella posizione della directory corrente la cartella PROGETTO_MCF.

---
# Obiettivo e fisica della simulazione

Il programma vuole implementare il comportamento delle particella poste in campi magnetici, dove sono presenti o dei campi elettrici o dei gradienti magnetici perpendicolari al campo magentico. 
La presenza di questi ultimi genera una velocità di deriva per la particella, descritta da:

$$v_d = \frac{E\times B}{B^2}$$

Nel caso del campo elettrico, oppure:

$$v_d = \pm\frac{1}{2}v_{perp}r_L\frac{B\times\nabla B}{B^2}$$

Nel caso del gradiente magnetico. 
La velocità di deriva devia il moto della particella rispetto al semplice moto ciclotronico che si avrebbe nel caso fosse presente solo il campo magnetico della particella. Ci si aspetta che la simulazione riesca dunque a riprodurre i risultati teorici. Per farlo vengono simulate miglaia di particelle con diverse configurazioni delle intensità di $E$ e di $\nabla B$, ci si aspetta che la velocità di deriva cresca linearmente con queste diverse configurazioni, considerando delle particelle cariche positivamente e negativamente e velocità iniziali casuali. Per maggiori informazioni riguardo la teoria dei moti di deriva consultare il capitolo 1 del file: Moti_di_deriva.pdf.

Per ricavare la traiettoria della particella nei due casi di drift, viene implementato l'algortimo di Boris che permette di descrivere come si muove una particella posta in campi elettrici e magnetici. Tramite poi il moto ciclotronico della particella si ricava la traiettoria del centro di guida, tramite una funzione che media su ogni orbita della particella e ne ricava il centro. Infine la velocità di deriva viene calcolata con una funzione che deriva semplicemente la velocità del centro di guida a partire dai punti ricavati precedentemente, a questo risultato viene tolto il contributo del moto ciclotronico ottenendo così la velcoità di deriva della particella simulata. Per maggiori informazioni sull'implementazione dell'algortimo della simulazione, consultare il capitolo 2 del file: Moti_di_deriva.pdf.

La capacità della simulazione, nel riprodurre dei risultati veritieri, è stata studiata tramite indagine statistica sui due diversi tipi di drift. Prima si ricavava il modulo della velocità media di deriva tramite simulazione per un campione di $1000$ particelle, verificando dunque l'accordo con la previsione teorica. Successivamente si è eseguito un fit lineare sulle velocità di deriva medie, con diverse configurazioni di campi elettrici o di gradienti magnetici, verificando la dipendenza lineare della velocità da questi. I risultati ricavati sono in accordo con la teoria. Per maggiori informazioni sull'analisi dati della simulazione consultare il capitolo 3 del file: Moti_di_deriva.pdf.

---
# Argomenti del programma

Il programma permete di essere seguito con degli argomenti tramite la libreria argparse. La lista completa degli argomenti utilizzabili è la seguente:

 * **drE**: Esegue la simulazione per il drift dato da $E \times B$
 
 * **drG**: Esegue la simulazione per il drift dato da $\nabla B$
 
 * **tra**: Esegue il programma in modalità traiettoria
 
 * **data**: Esegue il programma in modalità analisi dati
 
 * **save**: Permette il salvataggio dei dati in modalità default
 
 * **clean**: Elimina, se presente, il file drift_data.csv

 * **step**: Permette di modificare il numero di step fatti da ogni particella (Default=$3000)
 
 * **turb**: Permette di simulare le turbolenze magnetiche (Default=$0.000$), il numero deve essere scelto nell'intervallo $[0.000,1.000]$
 
L'utilizzo corretto dei vari argomenti verrà specificato nelle sezioni successive.
 
---
# Struttura e utilizzo del programma

Il programma ha tre diverse modalità di esecuzione, ognuna in base al tipo di drift scelto:

 * **Default**: Modalità in cui la funzione esegue la simulazione per $1000$ particelle con velocità iniziali e cariche casuali. L'utente dovrà configurare il valore di $B_z$ e delle componente del campo $E$ o $\nabla B$ in base al tipo di drift scelto. Il programma restituisce l'analisi dati del modulo della velocità media di deriva con errore e confronto teorico. I dati ricavati in questa modalità possono essere salvati tramite argomento `--save`. 
 Non serve nessun argomento per attivare la modalità ma basta solo scegliere il tipo di drift.
 Esempio di utilizzo della modalità di default, nella directory `/Simulazione`, digitare:
 ```bash
 Python3 main.py --drE --save
 ```
 Il comando esegue la simulazione in modalità default per il drift $E\times B$, in più al termine del processo salva i dati nel file drift_data.csv.
 
 * **Traiettoria**: Modalità in cui il progrmma ricava le traiettorie di $5$ particelle con il drift scelto come argomento. Anche qui si deve configurare il campo $E$ o $\nabla B$ e $B_z$. Il risultato della simulazione mostrerà i plot dell traiettorie e alcuni informazioni sui parametri scelti per la simulazione e i valori delle velocità di deriva ricavata confrontati con i valori teorici attesi. Questa modalità è utile per provare diverse configurazioni di campi, $B_z$, numero di passi etc. Se infatti la simulazione non fornirà dei risultati soddisfacenti, si potranno configurare meglio prima di prendere e salvare i dati tramite modalità default.
 I dati ricavati in questa modalità non possono essere salvati.
 L'argomento per attivare la modalità default è: `--tra`.
 Esempio di utilizzo della modalità traiettoria, nella directory `/Simulazione`, digitare:
 ```bash
 Python3 main.py --drG --tra
 ``` 
 Il comando esegue la simulazione per il drift $\nabla B$ in modalità traiettoria con $5$ particelle.
 
 * **Analisi dati**: Modalità in cui il programma non esegue la simulazione dei moti di deriva ma calcola soltanto il fit lineare per il drift scelto, sulla base dei dati salvati nel file: drift_data.csv. L'utente non deve configurare nulla ma dovrà accertarsi dell'esistenza nella cartella `/Simulazione` del file dei dati e che questo ne contenga qualcuno. I dati provengono dalla modalità di default, basta quindi eseguire la simulazione per un drift in questa modalità, con l'argomento `--save` per creare il file se non dovesse essere presente.
 Il risultato della modalità e la stampa del fit lineare dei dati con le relative informazioni.
 I dati ricavati in questa modalità non possono essere salvati.
 L'argomento per attivare la modalità default è: `--data`.
 ```bash
 Python3 main.py --drG --data
 ``` 
 Il comando esegue la simulazione per il drift $\nabla B$ in modalità analisi dati.
 
---
# Configurazione dei parametri e range consentiti

Una volta avviato il programma in modalità default o traiettoria e scelto il tipo di drift che si vuole simulare, l'utente dovrà configurare i valori dei campi o del gradiente. Per come il programma è stato scritto, non tutti i valori per $B_z$ e dei campi genera un risultato veritiero. Alcuni valori troppo bassi o troppo alti possono generare errori o restituire risultati poco corretti. Le condizioni che permettono di ottenere dei risultati validi sono trattate in dettaglio nel capitolo 2 del file Moti_di_deriva.pdf. Di seguito vengono elencati solo i range consentiti che permettono di ricavare dei risultati coerenti per la simulazione scelta:

 * **$B_z$** Range consentito: $1e^{-3} - 9e^{-4}$ T
 
 * **$E_x$, $E_y$** Range consigliato: sull'ordine delle decine di V/m
 
 * **$E_z$** Range consigliato: sull'ordine delle decine di V/m e si consiglia di non sceglerlo troppo altro rispetto alle altre componenti per non coprire eccessivamente il moto di deriva
 
 * **$\nabla B$** Range consentito: $1e^{-7} - 9e^{-7}$ T/m

Quando viene eseguito il programma, vengono stampati dei valori consigliati per ottenere dei risultati chiari. Si può comunque utilizzare la modalità traiettoria per visualizzare bene la configurazione scelta e verificare che restituisca risultati corretti.

 * Utilizzando l'argomento `--turb`, si deve inserire un valore compreso tra $[0.000,1.000]$ come già accennato. Valori anche di poco alti del coefficiente comportano a enormi deviazioni della traiettoria e risultati lontani dalla previsione teorica. 

 * L'argomento `--step` permette di modificare il numero di passi della simulazione. Tuttavia, un valore troppo alto rende la simulazione molto lenta mentre un valore troppo basso rompe la simulazione in quanto non riuscirebbe a calcolare la velocità di deriva. Si consiglia di utilizzare un numero di passi tali da compiere almeno un numero maggiore di $20$ orbite. Il numero di passi per compiere un orbita è descritto dalla relazione:
 
 $$N=\frac{2\pi m}{qBdt}$$

 dove $dt=1\cdot10^{-6}s$.

---
# Esempi di utilizzo

Vengono mostrati alcuni esempi di utilizzo del programma

```bash
Python3 main.py -G -T -N 5000
```

Esegue la simulazione in modalità traiettoria per il drift del gradiente magnetico con $5000$ passi per ogni particella.

```bash
Python3 main.py -G -N 5000 -t 0.005 -s
```

Esegue il programma in modalità default per il drift del campo elettrico con turbolenza pari a $0.005$ (bassa) e salva i dati della simulazione.

```bash
Python3 main.py -c -E
```

Viene eliminato il file dei dati e l'argomento `-E` non viene considerato dal programma.

```bash
Python3 main.py -d -E
```

Esegue il programma in modalità analisi dati per il drift del campo elettrico

# Avvertenze

Alcune combinazioni di argomenti potrebbero portare a un errato utilizzo del programma e pertanto questo non funzionerà. 

 * Scegliere come argomento sia `--drE` che `--drG`, non può essere fatto, si deve infatti scegliere un solo tipo di drift. In generale, la scelta di un solo drift è obbligatoria per ognuna delle modalità di esecuzione del programma.

 * Non è possibile salvare i risultati per la modalità traiettoria o analisi dati, il salvataggio viene fatto solo per i dati della modalità di default poiché sono necessari per la parte dell'analisi dati.
 
 * L'argomento `--turb` e `--step` funziona non funziona per la parte di analisi dati.
 
 * **!MOLTO IMPORTANTE!**: Per avere un'analisi dati ottimale è necessario avere lo stesso valore per tutte le misurazioni effettuate del valore $B_z$ e del numero di passi. Quando vengono salvati i dati della simulazione del file e vengono successivamente letti dalla funzione apposita in modalità analisi dati, questa si accerterà della seguente condizione. Se i dati avranno valori diversi per questi parametri non sarà possibile eseguire l'analisi dati per il drift selezionato. Sarà necessario o cancellare le righe interessate dal file o cancellare il file direttamente tramite l'argomento apposito. Si consiglia pertanto di prestare attenzione quando si salvano i dati della simulazione di avere sempre $B_z$ e il numero di passi uguale per ogni siumulazione del drift scelto.
 
 * Il file drift_data.csv contiene già $5$ misurazioni per ogni tipo di drift con diverse configurazioni di campi. Si può utilizzare da subito l'argomento --data. Se si vogliono prendere altre misurazioni si consiglia di utilizzare $B_z=8e-4$ e $step = 3000$, altrimenti `--data` non funzionerà.
 
 * Per eseguire il programma si deve utilizzare solo il file main.py, gli altri script servono e vengono utilizzati da quest'ultimo e non sono da eseguire. 
