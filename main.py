from loader import carica_dataset
from baseline_grezzo import esegui_scenario_base
from kdd_ottimizzato import esegui_scenario_kdd


def main():
    print("--- ANALISI DATA MINING: TELCO CHURN ---")

    # 1. Caricamento
    df = carica_dataset('telco.csv')

    # 2. Baseline
    acc_a, cm_a = esegui_scenario_base(df)

    # 3. KDD
    acc_b, cm_b = esegui_scenario_kdd(df)

    # 4. Confronto
    print("\n--- RISULTATI FINALI ---")
    print(f"Accuratezza Grezza:      {acc_a:.2%}")
    print(f"Accuratezza Ottimizzata: {acc_b:.2%}")

    delta = acc_b - acc_a
    if delta > 0:
        print(f"Miglioramento ottenuto: +{delta:.2%}")
    else:
        print(f"Differenza: {delta:.2%}")

    # L'accuratezza da sola dice poco su un dataset sbilanciato (~73% No / ~27% Sì):
    # un modello che indovinasse sempre "No" prenderebbe già circa il 73%. Il
    # report di classificazione stampato sopra per ciascuno scenario mostra il
    # recall sulla classe "Sì" (quanti abbandoni reali vengono davvero
    # individuati), che racconta la storia più interessante del miglioramento.

    print("\nGenerazione grafici consigliata: eseguire 'grafici.py'")


if __name__ == "__main__":
    main()
