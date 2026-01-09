import numpy as np
import plots as pt
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit


def vd_fit(v_drift, v_drift_th):
    
    """
    Funzione che calcola e stampa la velocità di drift media e la deviazione standard
    Vengono calcolate le componenti medie e la deviazione standard delle velocità di drift
    Si ricava dunque il modulo della velocità di drift media e il suo errore da confrontare con la teoria
    
    Parametri:
    ----------
    v_drift    : Array delle velocità di drift ricavate per ogni particella [m/s]
    v_drift_th : Array delle velocità di drift teoriche per ogni particella [m/s]

    Ritorna:
    --------
    vd_mean    : Valore medio del modulo della velocità di drift [m/s]
    vd_err     : Errore associato al valore medio del modulo della velocità di drift [m/s]
    vd_th_mean : Valore teorico medio del modulo della velocità di drift [m/s]
    """

    mu = np.zeros(2)      # Media delle componenti della velocità di drift
    sigma = np.zeros(2)   # Deviazione standard delle componenti della velocità di drift
    
    # Fit gaussiano delle componenti della velocità di drift
    components = ['x', 'y']
    for i, comp in enumerate(components):
        mu[i], sigma[i] = norm.fit(v_drift[:,i]) 

    # Calcolo della velocità di drift media e errore tramite propagazione
    vd_mean = np.linalg.norm(mu, axis=0) 
    vd_err = sigma / np.sqrt(len(v_drift))
    vd_err_final = np.sqrt((mu[0] / vd_mean * vd_err[0] )**2 + (mu[1] / vd_mean * vd_err[1])**2)
    
    # Calcolo della velocità di drift teorica media
    vd_th_mean_vec = np.mean(v_drift_th, axis=0)
    vd_th_mean = np.linalg.norm(vd_th_mean_vec)

    # Calcolo dell'errore relativo tra simulazione e teoria
    rel_err = np.abs((vd_mean - vd_th_mean) / vd_th_mean) * 100

    # Genera il grafico delle distribuzioni delle componenti della velocità di drift
    pt.plots_vd_dist(v_drift, vd_th_mean_vec, mu, sigma)
    
    # Stampa dei risultati  
    print(f"\n-------------------------------------------------------------")
    print(f"Analisi della velocità di drift con {len(v_drift)} particelle:\n")
    
    for i, comp in enumerate(components):
        print(f"Componente {comp} della velocità di drift media:    {mu[i]:.2f} ± {vd_err[i]:.2f} [m/s]")
        print(f"Deviazione standard componente {comp}:              {sigma[i]:.2f} [m/s]")
        print(f"Componente {comp} teorica della velocità di drift:  {vd_th_mean_vec[i]:.2f} [m/s]\n")

    print(f"Velocità di drift media:        {vd_mean:.2f} ± {vd_err_final:.2f} [m/s]")
    print(f"Velocità di drift teorica:      {vd_th_mean:.2f} [m/s]")
    print(f"Errore relativo:                {rel_err:.2f} %\n") 
    
    return vd_mean, vd_err_final, vd_th_mean


def linear_fit(fields_value, v_drift_mean, v_drift_err):

    """
    Funzione che calcola e stampa il fit lineare delle velocità medie con la previsione teorica
    Il fit è ricavato attraverso diverse configurazioni di campo dove è ricavato il modulo della componente perpendicolare al campo B
   
    Parametri:
    ----------  
    fields_value : Array con misure dei moduli dei campi delle misurazioni
    v_drift_mean : Array con misure dei moduli delle velocità medie di deriva
    v_drift_err  : Array con errori associati alle velocità medie

    Ritorna:
    --------
    m_fit : Coefficiente angolare del fit lineare
    m_err : Errore associato al coefficiente angolare
    """

    popt, pcov = curve_fit(linear_func, fields_value, v_drift_mean, sigma=v_drift_err, absolute_sigma=True)
    m_fit = popt[0]
    m_err = np.sqrt(pcov[0,0])

    return m_fit, m_err


def linear_func(x, m):

    """
    Funzione lineare per il fit delle velocità di drift
    
    Parametri:
    ----------
    x : Valore del modulo del campo perpendicolare a B (fields_value)
    m : Coefficiente angolare del fit
    
    Ritorna:
    -------- 
    mx : Valore della velocità di drift calcolata dalla funzione lineare
    """    
    
    mx = m * x
    
    return mx


def select_data(df, flag):
    
    """
    Funzione che seleziona i dati del dataframe in base al tipo di drift scelto dall'utente
    Per avere una migliore analisi dati controlla che Bz e N_steps non siano diversi per le prese dati

    Parametri:
    ----------
    df   : DataFrame contenente i dati delle simulazioni

    Ritorna:
    --------
    data : Array con i dati selezionati in base al tipo di drift 
    """
    
    # Scelta dei dati in base al drift
    df_data = df[df['Flag'] == flag]

    #--------------------------------------------------------------
    # Controllo che i dati abbiano lo stesso Bz e N_steps o non siano vuoti
    
    Bz_check = df_data['Bz'].nunique() 
    N_steps_check = df_data['N_steps'].nunique() 
    
    if Bz_check == 1 and N_steps_check == 1:
        
        data = df_data[["v_drift", "v_drift_err", "v_drift_theor", "Fields_value", "Bz", "Turbulence_coeff"]].to_numpy()
    
    elif df_data.empty:
        
        print(f"\nNessun dato trovato per il drift {flag}\n")
        return None
    
    else:
        
        if not Bz_check == 1:
            
            print("\nAttenzione: Il drift contiene diverse configurazioni di campo\n") 
            return None

        else:
            
            print("\nAttenzione: Il drift contiene diverse configurazioni di passi\n")
            return None
    #--------------------------------------------------------------
    
    return data