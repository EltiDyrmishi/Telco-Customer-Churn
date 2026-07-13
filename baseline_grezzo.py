import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

RANDOM_STATE = 42
TEST_SIZE = 0.3
K_BASELINE = 25  # scelto a caso, senza alcun tuning: è proprio quello che un
                 # approccio "grezzo" farebbe, e fa parte del problema che voglio dimostrare


def esegui_scenario_base(df_input):
    print("\n[2] Esecuzione Scenario BASELINE (Dati Grezzi)")
    print("    --------------------------------------------")

    # Lavoro su una copia per non toccare l'originale
    df = df_input.copy()

    # --- IL "SABOTAGGIO" ---
    # Qui faccio apposta a lasciare 'customerID'. Lo trasformo solo in numero.
    # Perché? Perché voglio dimostrare la "Maledizione della Dimensionalità".
    # Il KNN penserà che il cliente n.100 e il n.101 sono "vicini", ma è solo un caso!
    # Questo è puro rumore che confonde l'algoritmo.
    if 'customerID' in df.columns:
        df['customerID'] = df['customerID'].astype('category').cat.codes
        print("    [!] ATTENZIONE: Mantengo 'customerID' trasformato in numero.")
        print("        Questo introduce rumore inutile per vedere se il modello sbaglia.")

    # Trasformo le variabili categoriche (testo) in numeri (0/1) altrimenti sklearn non va.
    # Uso drop_first=True come nello scenario KDD: così il confronto tra i due scenari
    # riguarda solo i dati (grezzi vs puliti), non anche uno schema di encoding diverso.
    df = pd.get_dummies(df, drop_first=True)

    # Separo il target (chi se ne va) dal resto dei dati
    if 'Churn_Yes' in df.columns:
        y = df['Churn_Yes']
        X = df.drop(columns=['Churn_Yes'])
    else:
        target = [c for c in df.columns if 'Churn' in c][0]
        y = df[target]
        X = df.drop(columns=[target])

    # Divido in Train e Test (70/30).
    # Uso 'stratify' per essere sicuro di avere la stessa % di abbandoni in entrambi i set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Anche il Baseline ha bisogno di gestire i valori mancanti di TotalCharges
    # (altrimenti KNN si rompe su quei NaN). La mediana la calcolo SOLO sul
    # training set e la riuso identica per riempire i buchi nel test set.
    mediana_train = X_train['TotalCharges'].median()
    X_train['TotalCharges'] = X_train['TotalCharges'].fillna(mediana_train)
    X_test['TotalCharges'] = X_test['TotalCharges'].fillna(mediana_train)

    # Addestro il KNN senza normalizzare i dati.
    # Questo è l'errore classico: il modello darà troppa importanza ai prezzi (numeri grandi)
    # e ignorerà i mesi di fedeltà (numeri piccoli).
    print(f"    -> Addestro il KNN (k={K_BASELINE}) sui dati sporchi e non scalati...")
    model = KNeighborsClassifier(n_neighbors=K_BASELINE)
    model.fit(X_train, y_train)

    # Faccio le previsioni
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"    -> Finito. Accuratezza ottenuta: {acc:.2%}")
    print("\n    [Report Classificazione Baseline]")
    print(classification_report(y_test, y_pred))

    return acc, cm
