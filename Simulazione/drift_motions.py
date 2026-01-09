import numpy as np


def drift(N, dt, B, E, B_grad, qm, v0, n_t):
   
    """
    Funzione che calola la traiettoria e la velocità di una particella con il drift scelto
    Implementa il metodo di Boris per l'integrazione 
    Si calcola il campo magnetico locale in base alla posizione della particella
   
    Parametri:
    ----------
    N   : Numero di passi della simulazione
    dt  : Intervallo di tempo tra i passi [s]
    B   : Campo magnetico di riferimento [T]
    E   : Campo elettrico di riferimento [V/m]
    B_grad : Gradiente del campo magnetico [T/m]
    qm  : Rapporto carica massa [C/Kg]
    v0  : Velocità iniziale della particella [m/s]
    n_t : Coefficiente di scattering

    Ritorna:
    --------
    r   : Array delle posizioni della particella ad ogni passo [m]
    v   : Array delle velocità della particella ad ogni passo [m/s]
    """

    # Inizializzazione arrray posizione e velocità
    r = np.zeros((N, 3))
    v = np.zeros((N, 3))
    v[0, :] = v0

    #------------------------------------------------------------
    # Moto della particella

    for n in range(N-1):

        # Randomizzazione direzione particella
        scatter = np.random.uniform(0.001,1.000)

        # Calcolo del campo magnetico locale
        B_loc = np.array([0.0, 0.0, B[2] + B_grad[0] * r[n,0] + B_grad[1] * r[n,1]])

        # Creazione dei vettori di rotazione s e t
        t = np.array([0.0, 0.0, qm * B_loc[2] * dt / 2.0]) 
        t2 = np.dot(t, t)
        s = 2 * t / (1 + t2)

        # v_minus
        v_minus = v[n] + qm * E * dt / 2

        # v_prime = v_minus + v_minus × t
        v_prime = v_minus + np.cross(v_minus, t)

        # v_plus = v_minus + v_prime × s
        v_plus = v_minus + np.cross(v_prime, s)

        # Variabile check per turbolenza
        v_new = v_plus + qm * E * dt / 2

        # Turbolenza
        if scatter < n_t:
            v_mod = np.linalg.norm(v_new)
            v[n+1] = v_mod * turbulence_effects()
        else:
            v[n+1] = v_new 

        # Aggiornamento posizione
        r[n+1] = r[n] + v[n+1] * dt
    #------------------------------------------------------------

    return r, v


def guide_center(r, n_orb, steps_orb):

    """
    Funzione che calcola la traiettoria del centro di guida 
    Media le posizioni della particella su ogni orbita e ricava la posizione del centro di guida

    Parametri:
    ----------
    r         : Array delle posizioni della particella [m]
    n_orb     : Numero di orbite del moto
    steps_orb : Numero di passi per orbita

    Ritorna:
    --------
    r_gc : Array con posizioni del centro di guida [m]
    """

    # Creazione array delle posizione del centro di guida
    r_gc = np.zeros((n_orb, 3))

    # Calcolo delle posizioni del centro di guida mediando su ogni orbita
    for i in range(n_orb):

        i0 = i * steps_orb
        i1 = (i + 1) * steps_orb
        r_gc[i] = np.mean(r[i0:i1], axis=0) 

    return r_gc


def v_drift(r_gc, n_orb, T_orb, B_hat):
    
    """
    Funzione che calcola la velocità di drift della particella
    Calcola la derivata del centro di guida per ricavare la velocità 
    Da quella calola la velocità di drift sottraendo la componente parallela a B

    Parametri:
    ----------
    r_gc   : Array delle posizioni del centro di guida [m]
    n_orb  : Numero di orbite del moto
    T_orb  : Periodo per compiere un orbita [s]
    B_hat  : Versore campo magnetico 

    Ritorna:
    --------
    v_d    : Valore della velocità di drift ricavata [m/s]
    v_d_th : Valore della velocità di drift teorica [m/s]
    """
    
    # Calcolo della velocità del centro di guida
    v_gc_vec = (r_gc[-1] - r_gc[0]) / (n_orb * T_orb)
    
    # Calcolo della velocità di drift
    v_d_vec = v_gc_vec - np.dot(v_gc_vec, B_hat) * B_hat

    return v_d_vec


def turbulence_effects():
  
    """
    Funzione che scattera la direzione della particella in una direzione casuale 
    Utilizza coordinate sferiche per generare i valori casuali della direzione
    
    Parametri:
    ----------
    Nessuno   

    Ritorna:
    --------
    rand_dir : Array per la direzione della particella con valori casuali
    """	

    # Definizioni dei valori che vengono generati casualmente per phi e theta
    cos_th = np.random.uniform(-1.0, 1.0)
    sin_th = np.sqrt(1-cos_th**2)
    phi = np.random.uniform(0.0, 2.0*np.pi)
    
    # Definizione dei versori
    dir_x = sin_th * np.cos(phi)
    dir_y = sin_th * np.sin(phi)
    dir_z = cos_th
    rand_dir = np.array([dir_x, dir_y, dir_z])
    
    return rand_dir