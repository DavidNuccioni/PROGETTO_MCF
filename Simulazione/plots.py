import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import analysis as an

def plots_tra(position, guide_cn):

    """
    Funzione che crea i plot delle traiettorie delle particelle e dei centri di guida
    Stampa inoltre informazioni riguardo la simulazione effettuata

    Parametri:
    ----------
    position  : Array delle posizioni delle particelle [m]
    guide_cn  : Array delle posizioni dei centri di guida [m]

    Ritorna:
    --------
    Nessuno
    """

    #--------------------------------------------------------------
    # Plot 3D delle traiettorie con drift E×B
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Lista dei colori per le traiettorie
    c = ['limegreen', 'olivedrab', 'chocolate', 'royalblue', 'darkorchid']
    
    # Disegno delle traiettorie e dei centri di guida
    for p, r in enumerate(position):
        ax.plot(r[:,0], r[:,1], r[:,2], lw=2, color=c[p], alpha=0.5, label=(f"Particella {p+1}"))
        ax.scatter(r[-1][0], r[-1][1], r[-1][2], color=c[p], s=30)
    
    for p, r_gc in enumerate(guide_cn):
        ax.plot(r_gc[:,0], r_gc[:,1], r_gc[:,2], lw=2, color=c[p])
    
    # Disegna l'origine delle traiettorie
    ax.scatter(r[0][0], r[0][1], r[0][2], color='red', s=30, label=(f'Origine'))

    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    ax.set_title('Traiettorie delle particelle')
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # Plot 2D con drift E×B
    
    fig2, ax2 = plt.subplots(figsize=(6,6))
    
    # Disegno delle traiettorie e dei centri di guida
    for p, r in enumerate(position):
        ax2.plot(r[:,0], r[:,1], lw=2, color=c[p], alpha=0.5, label=(f'Particella {p+1}'))
        ax2.scatter(r[-1][0], r[-1][1], color=c[p], s=30)
   
    for p, r_gc in enumerate(guide_cn):
        ax2.plot(r_gc[:,0], r_gc[:,1], lw=2, color=c[p])
   
    # Disegna l'origine delle traiettorie
    ax2.scatter(0.0, 0.0, color='red', s=30, label=(f'Inizio'))

    ax2.set_xlabel('x[m]')
    ax2.set_ylabel('y[m]')
    ax2.set_title('Proiezione 2D delle traiettorie')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    #--------------------------------------------------------------
    
    return


def plots_vd_dist(v_drift, v_drift_th, mu, sigma):

    """
    Funzione che disegna i plot delle distribuzioni delle componenti della velocità di drift
    Viene mostrato il paragona tra la distribuzione e il fit sulla gaussiana

    Parametri:
    ----------
    v_drift    : Array delle velocità di drift ricavate per ogni particella [m/s]
    v_drift_th : Array delle velocità di drift teoriche per ogni particella [m/s]
    mu         : Array delle medie delle componenti della velocità di drift [m/s]
    sigma      : Array delle deviazioni standard delle componenti della velocità di drift [m/s]

    Ritorna:
    --------
    Nessuno
    """

    components = ['x', 'y']
    fig, axs = plt.subplots(1,2, figsize=(12,5))

    for i, comp in enumerate(components):
        
        # Genera istogramma delle componenti della velocità di drift
        axs[i].hist(v_drift[:,i], bins=50, density=True, alpha=0.6, color='skyblue', label='Simulazione')
        
        # Disegna la curva gaussiana del fit
        x_vals = np.linspace(min(v_drift[:,i]), max(v_drift[:,i]), 200)
        axs[i].plot(x_vals, norm.pdf(x_vals, mu[i], sigma[i]), '--', color='red', label=(f'Fit Gaussiano μ={mu[i]:.1f}'))
        
        # Disegna una linea della media teorica
        axs[i].axvline(v_drift_th[i], color='green', alpha=0.6, linestyle='-', linewidth=2, label=(f'Valore teorico v={v_drift_th[i]:.1f} m/s'))
        
        axs[i].set_xlabel(f'Componente {comp} [m/s]')
        axs[i].set_ylabel(f'PDF')
        axs[i].legend()

    plt.suptitle('Distribuzioni componenti della velocità di drift')
    plt.tight_layout()
    plt.show(block=False)

    return


def plots_vd_fit(fields_value, v_drift_mean, v_drift_err, m_fit, m_err, m_th, v_drift_th):

    """
    Funzione che disegna il plot del fit lineare delle velocità medie con la previsione teorica

    Parametri:
    ----------
    fields_value  : Array dei valori caratteristici del campo [V/m o T/m]
    v_drift_mean  : Array delle velocità di drift medie ricavate per ogni configurazione [m/s]
    v_drift_err   : Array degli errori delle velocità di drift medie [m/s]
    lin_fuc       : Funzione lineare del fit presente in analysis.py
    m_fit         : Coefficiente angolare del fit
    m_err         : Errore sul coefficiente angolare del fit
    m_th          : Valore teorico del coefficiente angolare del fit
    v_drift_th    : Array delle velocità di drift teoriche per ogni configurazione [m/s]

    Ritorna:
    --------
    Nessuno
    """

    plt.figure(figsize=(8,6))
    
    # Plot dei dati con barre d'errore
    plt.errorbar(fields_value, v_drift_mean, yerr=v_drift_err, fmt='o', label='Dati Simulazione', color='blue', ecolor='blue', elinewidth=3, capsize=3)
    
    # Plot della retta di fit
    x_fit = np.linspace(0, max(fields_value)*1.1, 100)
    y_fit = an.linear_func(x_fit, m_fit)
    plt.plot(x_fit, y_fit, '-', color='red', label=(f'Fit Lineare: m = {m_fit:.2e} ± {m_err:.2e}'))
    
    # Plot della retta teorica
    y_th = m_th * x_fit
    plt.plot(x_fit, y_th, '--', color='green', label=(f'Teoria: m = {m_th:.2e}'))  
    
    # Plot dei dati teorici
    plt.plot(fields_value, v_drift_th, 'x', markersize=8, markeredgewidth=2 , color='black', label='Dati Teorici')

    plt.xlabel('Valore caratteristico del campo [V/m o T/m]')
    plt.ylabel('Velocità di drift media [m/s]')
    plt.title('Fit lineare della velocità di drift media')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    return