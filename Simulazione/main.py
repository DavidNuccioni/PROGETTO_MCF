import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm  
import drift_motions as dm
import analysis as an
import plots as pt


def parser_arguments():
    
    """
    Funzione che definisce gli argomenti da passare quando si esegue il codice
    """

    parser = argparse.ArgumentParser(description='Simulazione dei moti di deriva, scegliere un moto di deriva: -E o -G', usage ='python3 main.py --option')
    
    parser.add_argument('-N', '--step', type=int, action='store', default=3000, help='Inserisci il numero di passi (Default: 3000)')
    parser.add_argument('-t', '--turb', type=float, action='store', default=0.000, help='Inserisci il coefficiente di turbolenza tra [1.000 ; 0.000] (Default: 0.00)')    
    parser.add_argument('-E', '--drE', action='store_true', help='Esegue simulazione per drift ExB')
    parser.add_argument('-G', '--drG', action='store_true', help='Esegue simulazione per drift ∇ B')
    parser.add_argument('-T', '--tra', action='store_true', help='Esegue la simulazione per 5 particelle e ne mostra la traiettoria')
    parser.add_argument('-d', '--data', action='store_true', help='Esegui l\'analisi dei dati nel file drift_data.csv (consultare README)')
    parser.add_argument('-s', '--save', action='store_true',  help='Se scelto salva i dati della simulazione corrente nel file: drift_data.csv')
    parser.add_argument('-c', '--clean', action='store_true', help='Cancella il file drift_data.csv (consultare README)')
    
    return parser.parse_args(args=None if sys.argv[1:] else ['--help'])


def save_data(vd_mean, vd_err_final, vd_th_mean, fields_val, flag, N, n_t, Bz):	
    
    """
    Crea un dataframe con i dati della simulazione e lo salva nel file drift_data.csv

    Parametri:
    ----------
    vd_mean      : Modulo della velocità di deriva media
    vd_err_final : Errore della velocità di deriva media
    vd_th_mean   : Velocità di deriva teorica
    flag         : Flag di controllo per tipo di drift scelto
    N            : Numero di passi
    n_t          : Coefficiente turbolenza
    Bz           : Componente z del campo magnetico

    Ritorna:
    --------
    Nessuno
    """
	
	# Creazione dataframe vuoto
    df = pd.DataFrame(data={ 
        'Flag'              : [flag],
        'v_drift'           : [vd_mean],
        'v_drift_err'       : [vd_err_final],
        'v_drift_theor'     : [vd_th_mean],
        'Fields_value'      : [fields_val],
        'Turbulence_coeff'  : [n_t],
        'Bz'                : [Bz],
        'N_steps'           : [N]
	})
	
	# Salvataggio del dataframe in file .csv
    file_exists = os.path.exists(file_data)
    df.to_csv(file_data, mode='a' if file_exists else 'w', index=False, header=not file_exists)
	
    return


def clean_file(file_data):
        
    """
    Funzione che elimina, se presente il file dei dati
    
    Parametri:
    ----------
    file_data : File contenenti i dati della simulaizone

    Ritorna:
    --------
    Nessuno
    """
    
    if os.path.exists(file_data):
        
        os.remove(file_data)
        print(f"\nFile '{file_data}' eliminato con successo.\n")
    
    else:
        
        print(f"\nErrore: il file '{file_data}' non esiste nella cartella corrente\n")

    return


def simulation():

    """
    Funzione principale del moto di deriva della particella
    Controlla gli argomenti passati dall'utente e imposta i parametri della simulazione
    Esegue la simulazione e richiama le funzioni per l'analisi e la visualizzazione dei dati
    """

    #--------------------------------------------------------------
    # Elimina il file dati e chiude il programma
    if args.clean:
        
        clean_file(file_data)
        
        return
    #--------------------------------------------------------------
    

    #--------------------------------------------------------------
    # Esegue solo l'analisi dati per ricavare la dipendenza della velocità dai campi

    if args.data:

        #--------------------------------------------------------------
        # Controllo scelte utente
        if args.drE==False and args.drG==False:
            print(f"\nErrore: scegliere uno dei due modi per il moto di deriva della particella\nUsare --help per informazioni\n")
            return
        
        if args.drE and args.drG:
            print(f"\nErrore: scegliere solo uno dei due modi per il moto di deriva della particella\nUsare --help per informazioni\n")
            return
        
        if args.save:
            print(f"\nErrore: non è possibile salvare i dati in modalità analisi dati\n")
            return

        # Legge il file dei dati se esiste altrimenti termina il programma
        try:
                df = pd.read_csv(file_data)
            
        except FileNotFoundError:
                
                print("\nIl file non è stato trovato, assicurati che esista nella cartella corrente")
                print("Eseguire la simulazione con salvataggio dei dati per creare il file\n")
                
                return
        #--------------------------------------------------------------

        
        #--------------------------------------------------------------
        # Esegue l'analisi dati in base al tipo di drift scelto    
        if args.drE:
            
            # Estrapola i dati in base al drift scelto
            flag = 'ExB'
            data = an.select_data(df, flag)
            
            # Esegue l'analisi dati per il fit lineare se non ci sono stati problemi con i dati del dataframe
            if data is None:
                return
            
            else:

                # Estrazione dei dati dall'array
                fields_value = data[:,3]
                v_drift_mean = data[:,0]
                v_drift_err = data[:,1]
                v_drift_th = data[:,2]
                Bz = data[0,4]
                n_t = data[:,5]
                
                # Calcolo del coefficiente m teorico
                m_th = 1 / Bz
                
                # Funzione che esegue il fit lineare e funzione che genera il grafico
                m_fit, m_err = an.linear_fit(fields_value, v_drift_mean, v_drift_err)
                pt.plots_vd_fit(fields_value, v_drift_mean, v_drift_err, m_fit, m_err, m_th, v_drift_th)

                # Stampa delle informazioni
                print(f"-------------------------------------------------------------")
                print(f"\nDati utilizzati per il fit lineare\n")
                for i, m in enumerate(v_drift_mean):
                    print(f"Misurazione {i+1}:") 
                    print(f"Modulo del campo perpendicolare a Bz = {fields_value[i]:.2e} [V/m]")       
                    print(f"Velocità di drift media =              {v_drift_mean[i]:.2f} ± {v_drift_err[i]:.2f} [m/s]")
                    print(f"Turbolenza =                           {n_t[i]:.2f}\n")
                print(f"-------------------------------------------------------------")
                print(f"Risultati del fit lineare delle velocità di drift:\n")
                print(f"Coefficiente angolare del fit:    {m_fit:.2f} ± {m_err:.2f} [m²/(V·s)]")
                print(f"Valore teorico del coefficiente:  {m_th:.2f} [m²/(V·s)]\n") 
                print(f"Errore relativo del coefficiente: {np.abs((m_fit - m_th) / m_th) * 100:.2f} %\n")

        if args.drG:
            
            # Estrapola i dati in base al drift scelto
            flag = 'gradB'
            data = an.select_data(df, flag)
            
            # Esegue l'analisi dati per il fit lineare se non ci sono stati problemi con i dati del dataframe
            if data is None:
                return
            
            else:
                
                # Estrazione dei dati dall'array
                fields_value = data[:,3]
                v_drift_mean = data[:,0]
                v_drift_err = data[:,1]
                v_drift_th = data[:,2]
                Bz = data[0,4]
                n_t = data[:,5]
                
                # Calcolo del coefficiente m teorico
                m_th = ( np.mean(v_drift_th) / Bz**2)
                
                # Funzione che esegue il fit lineare e funzione che genera il grafico
                m_fit, m_err = an.linear_fit(fields_value, v_drift_mean, v_drift_err)
                pt.plots_vd_fit(fields_value, v_drift_mean, v_drift_err, m_fit, m_err, m_th, v_drift_th)

                # Stampa delle informazioni
                print(f"-------------------------------------------------------------")
                print(f"\nDati utilizzati per il fit lineare\n")
                for i, m in enumerate(v_drift_mean):
                    print(f"Misurazione {i+1}:") 
                    print(f"Modulo del campo perpendicolare a Bz = {fields_value[i]:.2e} [T/m]")       
                    print(f"Velocità di drift media =              {v_drift_mean[i]:.2f} ± {v_drift_err[i]:.2f} [m/s]")
                    print(f"Turbolenza =                           {n_t[i]:.2f}\n")
                print(f"-------------------------------------------------------------")
                print(f"Risultati del fit lineare delle velocità di drift:\n")
                print(f"Coefficiente angolare del fit:    {m_fit:.2e} ± {m_err:.2e} [m³/(T²·s)]")
                print(f"Valore teorico del coefficiente:  {m_th:.2e} [m³/(T²·s)]\n")
                print(f"Errore relativo del coefficiente: {np.abs((m_fit - m_th) / m_th) * 100:.2f} %\n")
        #--------------------------------------------------------------
        
        # Mantiene il grafico aperto prima della chiusura del programma
        plt.show()
    #--------------------------------------------------------------

        return


    #--------------------------------------------------------------
    # Controllo della scelta del tipo di drift e n_t per la simulazione
    
    if n_t > 1.0 or n_t < 0.0:
        print(f"\nErrore: scegliere un coeffieciente per la turbolenza compreso tra [1;0]\nUsare --help per informazioni\n")
        return
    
    if args.drE and args.drG:
        print(f"\nErrore: scegliere solo uno dei due modi per il moto di deriva della particella\nUsare --help per informazioni\n")
        return
    
    if args.drE: 
        
        # Flag per salvataggio dati
        flag = 'ExB'
        print(f"\nSimulazione del moto di deriva dato da E x B con turbolenza {n_t}")

    if args.drG:
        
        # Flag per salvataggio dati
        flag = 'gradB'
        print(f"\nSimulazione del moto di deriva dato dal ∇ B con turbolenza {n_t}")
    
    if args.drE==False and args.drG==False:
        print(f"\nErrore: scegliere uno dei due modi per il moto di deriva della particella\nUsare --help per informazioni\n")
        return
    #--------------------------------------------------------------

    
    #--------------------------------------------------------------
    # Configurazione dei campi scelta dall'utente

    # Parametri della simulazione per il drift ExB
    if args.drE:
        
        print(f"\n-------------------------------------------------------------")
        print(f"Configurazione iniziale dei campi E e B\n")
        
        # Componente z campo magnetico [T]
        try:
            Bz_inp = input(f"Inserire un valore per Bz [T] (consigliato: 8e-4): " )
            Bz = float(Bz_inp) 

        except (ZeroDivisionError, ValueError) :	
            print(f"\nErrore: valore non valido, deve essere un numero reale e non nullo\n")
            return          
        
        # Componente x campo elettrico [V/m]
        try:
            Ex_inp = input(f"Inserire un valore per Ex [V/m] (consigliato: ⁓1e1): " )
            Ex = float(Ex_inp)  
        
        except ValueError:	
            print(f"\nErrore: valore non valido, deve essere un numero reale\n")
            return
                
        # Componente y campo elettrico [V/m]
        try:
            Ey_inp = input(f"Inserire un valore per Ey [V/m] (consigliato: ⁓1e1): " )
            Ey = float(Ey_inp) 
        
        except ValueError:	
            print(f"\nErrore: valore non valido, deve essere un numero reale\n")
            return
        
        if Ey == 0.0 and Ex == 0.0:
            print(f"\nErrore: almeno una delle due componenti del campo elettrico deve essere diversa da zero\n")
            return

        # Componente z campo elettrico [V/m]
        try:
            Ez_inp = input(f"Inserire un valore per Ez [V/m] (consigliato: ⁓1e1): " )
            Ez = float(Ez_inp) 

        except ValueError:	
            print(f"\nErrore: valore non valido, deve essere un numero reale\n")
            return

        # Creazione array dei campi e del modulo di ExB
        E = np.array([Ex, Ey, Ez])
        B = np.array([0.0, 0.0, Bz])
        fields_val = np.linalg.norm(E[:2])

    # Parametri della simulazione per il drift gradB
    elif args.drG:

        print(f"\n-------------------------------------------------------------")
        print(f"Configurazione iniziale del campo B e di ∇ B\n")
        
        # Componente z campo magnetico [T]
        try:  
            Bz_inp = input(f"Inserire un valore per Bz [T] (consigliato: 8e-4): " )
            Bz = float(Bz_inp) 

        except (ZeroDivisionError, ValueError):	
            print(f"\nErrore: valore non valido, deve essere un numero reale e non nullo\n")
            return

        # Gradiente del campo magnetico sull'asse x [T/m]
        try:    
            dBdx_inp = input(f"Inserire un valore per ∇ B su x [T/m] (consigliato: ⁓e-7): " )         
            dBdx = float(dBdx_inp)

        except ValueError:	
            print(f"\nErrore: valore non valido, deve essere un numero reale\n")
            return
        
        # Gradiente del campo magnetico sull'asse y [T/m]
        try:    
            dBdy_inp = input(f"Inserire un valore per ∇ B su y [T/m] (consigliato: ⁓e-7): " )         
            dBdy = float(dBdy_inp)

        except ValueError:	
            print(f"\nErrore: valore non valido, deve essere un numero reale\n")
            return
        
        if dBdx==0 and dBdy==0:
            print(f"\nErrore: almeno una delle due componenti del gradiente deve essere diversa da zero\n")
            return

        # Creazione array del campo magnetico, del gradiente e del modulo gradBxB
        B = np.array([0.0, 0.0, Bz])
        B_grad = np.array([dBdx, dBdy, 0.0])
        fields_val = np.linalg.norm(B_grad[:2])
    #--------------------------------------------------------------


    #-------------------------------------------------------------- 
    # Parametri per centro di guida e velocità di drift
    
    try: 
        # Calcolo del periodo di ciclotrone
        B_mod = np.linalg.norm(B)       # Modulo del campo magnetico [T]
        B_hat = B / B_mod               # Versore del campo magnetico 
        om_c  = qm * B_mod              # Frequenza di ciclotrone [rad/s]
        T_c   = 2 * np.pi / om_c        # Periodo di ciclotrone [s]

        # Calcolo del numero di orbite
        steps_orb = int(T_c / dt)       # Passi per completare un orbita
        T_orb = steps_orb * dt          # Periodo per completare un orbita [s]
        n_orb = int(N / steps_orb)      # Numero di orbite completate

        # Creazioni di array per contenere i dati 
        position = np.zeros((N_par, N, 3))
        velocity = np.zeros((N_par, N, 3))
        guide_cn = np.zeros((N_par, n_orb, 3))
        v_drift  = np.zeros((N_par, 3))
        v_drift_th = np.zeros((N_par, 3))
        velocity_0 = np.zeros((N_par, 3))
        r_Larmor = np.zeros(N_par)
    
    except ZeroDivisionError:
        
        print(f"\nErrore: il campo magnetico ha un valore non corretto o i passi sono insufficienti\n")
        return
    #--------------------------------------------------------------
        

    #--------------------------------------------------------------
    # Inizio della simulazione
    
    try:
        #--------------------------------------------------------------
        print(f"\n-------------------------------------------------------------")
        print(f"Inizio della simulazione\n")
        print(f"Completamento del processo per {N_par} particelle...")
        for p in tqdm(range(N_par)):
            
            # Inizializzazione della carica per la particella
            sign = np.sign(np.random.uniform(-1.0, 1.0))
            qm_tra = sign * qm      
            
            # Creazione array per la velocità iniziale casuale della particella 
            v0x = np.random.normal(0.0 , 4e5)
            v0y = np.random.normal(0.0 , 4e5)
            v0z = np.random.normal(0.0 , 5e4)
            v0 = np.array([v0x, v0y, v0z])

            # Calcolo del raggio di Larmor della particella
            v_perp = np.linalg.norm(v0[:2])     # Componente perpendicolare della velocità iniziale [m/s]
            r_L = v_perp / om_c                 # Raggio di Larmor [m]

            #--------------------------------------------------------------
            # Esegue la simulazione per drift ExB

            if args.drE:
                
                # Calcolo della traiettoria e della velocità della particella
                r, v = dm.drift_EXB(N, dt, B, E, qm_tra, v0, n_t)
    
                # Calcolo della traiettoria del centro di guida
                r_gc = dm.guide_center(r, n_orb, steps_orb)

                # Calcolo della velocità di drift teorica per ExB
                v_d_th_vec = np.cross(E, B) / B_mod**2      # Velocità di drift teorica [m/s]
            #--------------------------------------------------------------


            #--------------------------------------------------------------
            # Esegue la simulazione per drift gradiente di B

            if args.drG:
                
                # Calcolo della traiettoria e della velocità della particella
                r,v = dm.drift_gradientB(N, dt, B, B_grad, qm_tra, v0, n_t)

                # Calcolo della traiettoria del centro di guida
                r_gc = dm.guide_center(r, n_orb, steps_orb)

                # Calcolo della velocità di drift teorica per gradB
                v_d_th_vec = ( v_perp**2 / (2 * qm_tra * B_mod**3)) * np.cross(B, B_grad)     
            #--------------------------------------------------------------
            
            # Calcolo della velocità di drift vettoriale della particella
            v_d_vec = dm.v_drift(r_gc, n_orb, T_orb, B_hat)
            
            # Salvataggio dei dati negli array
            position[p] = r
            velocity[p] = v
            guide_cn[p] = r_gc
            v_drift[p] = v_d_vec
            v_drift_th[p] = v_d_th_vec
            r_Larmor[p] = r_L 
            velocity_0[p] = v0
        #--------------------------------------------------------------
        print(f"\nSimulazione completata!")  
    
    except (IndexError or ValueError or ZeroDivisionError):
       
       print(f"\nErrore durante la simulazione: Inserire un numero maggiore di passi o valore per campo magnetico coerente\n")
       return   
    #--------------------------------------------------------------
    

    #--------------------------------------------------------------
    # Stampa dei risultati per modalità traiettoria

    if args.tra:
        
        pt.plots_tra(position, guide_cn)
 
        print(f"\n---------------------------------------------")
        print(f"Riepilogo della simulazione\n")

        print(f"Valore dei campi usati:")
        print(f"Bz = {Bz:.2e} [T]")
        
        if args.drE:

            E_str = ", ".join(f"{comp:.2f}" for comp in E)
            print(f"E = [{E_str}] [V/m]")      
        if args.drG:

            B_str = ", ".join(f"{comp:.2e}" for comp in B_grad)
            print(f"∇ B = [{B_str}] [T/m]")
        
        print(f"\nNumero di passi per particella: {N}")
        print(f"Numero di orbite:                 {n_orb}")
        print(f"Coefficiente di turbolenza:       {n_t:.3f}")

        print(f"\n---------------------------------------------")
        print(f"Esito della simulazione\n")
        
        for i in range(N_par):
            
            v_str = ", ".join(f"{comp:.2e}" for comp in velocity_0[i])
            print(f"Particella {i+1}:")
            print(f"Velocità iniziale: [{v_str}] [m/s]")
            print(f"Raggio di Larmor:            {r_Larmor[i]:.2f} [m]")
            print(f"Velocità di drift calcolata: {np.linalg.norm(v_drift[i]):.2f} [m/s]")
            print(f"Velocità di drift teorica:   {np.linalg.norm(v_drift_th[i]):.2f} [m/s]")
            print(f"Differenza percentuale:      {abs(np.linalg.norm(v_drift[i]) - np.linalg.norm(v_drift_th[i])) / np.linalg.norm(v_drift_th[i])*100:.2f} %\n")
        
        # Mantiene aperti i grafici prima della chiusura il programma
        plt.show() 

        if args.save:

            print(f"Errore: non è possibile salvare i file in modalità traiettoria\n")
        return 
    #--------------------------------------------------------------
    
   
    #--------------------------------------------------------------
    # Fit e informazioni della simulazione per 1000 particelle
    
    print(f"\n-------------------------------------------------------------")
    print(f"Riepilogo della simulazione\n")
    print(f"Valore dei campi usati:")
    print(f"Bz = {Bz:.2e} [T]")
    
    if args.drE:
        E_str = ", ".join(f"{comp:.2f}" for comp in E)
        print(f"E = [{E_str}] [V/m]")      
    
    if args.drG:
        B_str = ", ".join(f"{comp:.2e}" for comp in B_grad)
        print(f"∇ B = [{B_str}] [T/m]")
    
    print(f"\nNumero di passi per particella: {N}")
    print(f"Numero di orbite:                 {n_orb}")
    print(f"Coefficiente di turbolenza:       {n_t:.3f}")

    # Calcola il fit delle velocità di drift e stampa i risultati
    vd_mean, vd_err_final, vd_th_mean = an.vd_fit(v_drift, v_drift_th)
    
    # Mantiene aperto il grafico prima della chiusura del programma
    plt.show()
    #--------------------------------------------------------------


    #--------------------------------------------------------------
    # Salvataggio dei dati della simulazione se richiesto e chiusura programma
    
    if args.save:
       
        save_data(vd_mean, vd_err_final, vd_th_mean, fields_val, flag, N, n_t, Bz)
        print(f"-------------------------------------------------------------")
        print(f"I dati della simulazione sono stati salvati nel file: {file_data}\n")
    #--------------------------------------------------------------


if __name__ == "__main__":
    
    # Richiamo della funzione degli argomenti
    args = parser_arguments()
    
    # Definizione del file dei dati
    file_data = 'drift_data.csv'
 
    #--------------------------------------------------------------
    # Costanti e parametri per la simulazione

    q = 1.6e-19         # Carica della particella [C]
    m = 1.67e-27        # Massa della particella [kg]
    qm = q / m          # Rapporto carica massa per particella positiva [C/Kg]
    dt = 1e-6           # Intervallo di tempo tra i passi [s]        
    n_t = args.turb     # Coefficiente della turbolenza 
    N = args.step       # Numero di passi per il moto della particella         
    
    # Numero di particelle simulate
    if args.tra:
        N_par = 5

    else:
        N_par = 1000                 
    #--------------------------------------------------------------
    
    # Esecuzione della simulazione
    simulation()