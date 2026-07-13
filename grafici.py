import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
from loader import carica_dataset
from baseline_grezzo import esegui_scenario_base
from kdd_ottimizzato import esegui_scenario_kdd

# Impostazioni grafiche
sns.set_theme(style="whitegrid")

# Colori riusati in più grafici: un posto solo da cui cambiarli
COLOR_FEDELE = '#3498db'
COLOR_CHURN = '#e74c3c'


def genera_grafici():
    print("--- [GRAFICI] Inizio generazione... ---")

    # 1. GESTIONE CARTELLA (Versione Anti-Crash)
    nome_cartella = "immagini_output"

    # Provo a pulire la cartella se esiste
    if os.path.exists(nome_cartella):
        try:
            shutil.rmtree(nome_cartella)
            print(f"-> Vecchia cartella '{nome_cartella}' rimossa.")
        except PermissionError:
            print(f"-> ATTENZIONE: La cartella '{nome_cartella}' è aperta o in uso.")
            print("   Non posso cancellarla, ma sovrascriverò i file vecchi.")
        except Exception as e:
            print(f"-> Errore generico nella pulizia: {e}")

    # Creo la cartella
    os.makedirs(nome_cartella, exist_ok=True)
    print(f"-> Cartella di destinazione pronta: '{nome_cartella}'")

    # 2. Recupero i dati
    df = carica_dataset('telco.csv')

    # Eseguo gli scenari
    acc_a, cm_a = esegui_scenario_base(df)
    acc_b, cm_b = esegui_scenario_kdd(df)

    # --- GRAFICO 1: ISTOGRAMMA ---
    plt.figure(figsize=(10, 5))
    sns.histplot(df['TotalCharges'], kde=True, color='orange', bins=30)
    plt.xlabel("Totale Addebitato ($)")
    plt.ylabel("Numero di Clienti")
    percorso = os.path.join(nome_cartella, "1_distribuzione_costi.png")
    plt.savefig(percorso, bbox_inches='tight')
    print(f"-> Salvato: {percorso}")
    plt.close()

    # --- GRAFICO 2: TORTA ---
    # Uso reindex per garantire l'ordine No/Sì nelle label, invece di affidarmi
    # all'ordine di value_counts() (che ordina per frequenza: oggi "No" è la
    # classe più numerosa quindi viene per prima, ma non è garantito che lo sia
    # sempre — con reindex l'accoppiamento label-fetta è corretto a prescindere).
    plt.figure(figsize=(6, 6))
    conteggio = df['Churn'].value_counts().reindex(['No', 'Yes'])
    plt.pie(conteggio, labels=['Clienti Fedeli (No)', 'Abbandoni (Sì)'],
            autopct='%1.1f%%', colors=[COLOR_FEDELE, COLOR_CHURN], startangle=90)
    percorso = os.path.join(nome_cartella, "2_sbilanciamento_classi.png")
    plt.savefig(percorso, bbox_inches='tight')
    print(f"-> Salvato: {percorso}")
    plt.close()

    # --- GRAFICO 3: CONFRONTO MATRICI ---
    etichette_churn = ['No', 'Sì']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
                annot_kws={"size": 16}, xticklabels=etichette_churn, yticklabels=etichette_churn)
    axes[0].set_title(f"Scenario A: GREZZO\nAccuratezza: {acc_a:.2%}", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predetto")
    axes[0].set_ylabel("Reale")

    sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False,
                annot_kws={"size": 16}, xticklabels=etichette_churn, yticklabels=etichette_churn)
    axes[1].set_title(f"Scenario B: KDD OTTIMIZZATO\nAccuratezza: {acc_b:.2%}", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predetto")
    axes[1].set_ylabel("Reale")

    percorso = os.path.join(nome_cartella, "3_confronto_matrici.png")
    plt.savefig(percorso, bbox_inches='tight')
    print(f"-> Salvato: {percorso}")
    plt.close()

    # --- GRAFICO 4: CONTRATTI ---
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Contract', hue='Churn', palette=[COLOR_FEDELE, COLOR_CHURN])
    plt.xlabel("Tipo di Contratto")
    plt.ylabel("Numero di Clienti")
    plt.legend(title='Abbandono', labels=['No (Fedele)', 'Sì (Perso)'])
    percorso = os.path.join(nome_cartella, "4_ottimizzazione_contratti.png")
    plt.savefig(percorso, bbox_inches='tight')
    print(f"-> Salvato: {percorso}")
    plt.close()

    # --- GRAFICO 5: MATRICE DI CORRELAZIONE ---
    df_corr = df.copy()

    df_corr['Churn_Num'] = df_corr['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # exclude='number' invece di include=['object']: su Pandas 3.x le colonne di
    # testo possono avere dtype 'str' anziché 'object', e select_dtypes(include=
    # ['object']) le include ancora solo "per compatibilità" con un FutureWarning
    # che segnala la rimozione futura di questo comportamento. Escludere i tipi
    # numerici prende le stesse colonne testuali senza dipendere da quale dei
    # due dtype venga usato.
    for col in df_corr.select_dtypes(exclude='number').columns:
        if col != 'Churn':
            df_corr[col] = df_corr[col].astype('category').cat.codes

    # TRADUZIONE
    df_corr = df_corr.rename(columns={
        'Churn_Num': 'Abbandono',
        'tenure': 'Mesi Fedeltà',
        'MonthlyCharges': 'Costo Mensile',
        'Contract': 'Tipo Contratto',
        'TechSupport': 'Supporto Tecnico'
    })

    cols_ita = ['Abbandono', 'Mesi Fedeltà', 'Costo Mensile', 'Tipo Contratto', 'Supporto Tecnico']
    final_cols = [c for c in cols_ita if c in df_corr.columns]

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr[final_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    percorso = os.path.join(nome_cartella, "5_correlazione.png")
    plt.savefig(percorso, bbox_inches='tight')
    print(f"-> Salvato: {percorso}")
    plt.close()

    print(f"\n--- COMPLETATO: Controlla la cartella '{nome_cartella}' ---")


if __name__ == "__main__":
    genera_grafici()
