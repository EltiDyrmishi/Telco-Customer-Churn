import pandas as pd


def carica_dataset(nome_file: str) -> pd.DataFrame:
    """Legge il CSV e sistema solo il tipo di 'TotalCharges' (da testo a numero).

    Non riempio i valori mancanti qui dentro: lo fa ogni scenario singolarmente,
    DOPO aver separato train e test. Se calcolassi la mediana su tutto il
    dataset (come facevo prima) e solo dopo facessi lo split, la mediana
    userebbe anche dati che finiscono nel test set: è un piccolo data leakage,
    evitabile spostando l'imputazione dopo lo split.
    """
    print(f"\n[1] Caricamento dati dal file '{nome_file}'...")

    df = pd.read_csv(nome_file)

    # 'TotalCharges' viene letta come testo perché per gli 11 clienti con
    # tenure=0 (clienti nuovissimi, non ancora fatturati) il campo è uno spazio
    # vuoto invece di un numero. pd.to_numeric lo converte in NaN.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    n_mancanti = df['TotalCharges'].isna().sum()
    print(f"    -> Dataset caricato. Righe: {df.shape[0]}, Colonne: {df.shape[1]}")
    if n_mancanti:
        print(f"    -> 'TotalCharges' ha {n_mancanti} valori mancanti: verranno "
              f"riempiti dentro ogni scenario, con la mediana calcolata solo sul training set.")

    return df
