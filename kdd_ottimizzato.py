import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 42
TEST_SIZE = 0.3
IQR_MOLTIPLICATORE = 1.5
COLONNE_DA_SCALARE = ['MonthlyCharges', 'TotalCharges', 'tenure']
VALORI_K_DA_TESTARE = range(5, 51, 5)  # per la ricerca del k migliore in cross-validation


def esegui_scenario_kdd(df_input):
    print("\n[3] Esecuzione Scenario KDD OTTIMIZZATO")
    print("    -------------------------------------")

    df = df_input.copy()

    # 1. FEATURE SELECTION (Slide 24)
    # Rimuovo 'gender' perché statisticamente poco rilevante per il churn.
    # Rimuovo anche 'customerID': è un identificativo praticamente unico per
    # ogni cliente, quindi se lo lasciassi passare tale e quale, il prossimo
    # step (get_dummies) lo trasformerebbe in migliaia di colonne — una per
    # cliente — cioè lo stesso identico rumore che il Baseline lascia apposta
    # dentro per dimostrare la Maledizione della Dimensionalità. Qui va tolto.
    print("    -> Step 1: Feature Selection...")
    colonne_inutili = ['gender', 'customerID']
    colonne_presenti = [c for c in colonne_inutili if c in df.columns]
    if colonne_presenti:
        df = df.drop(columns=colonne_presenti)
        print(f"       Rimosse colonne non significative: {colonne_presenti}")

    # 2. BINNING DIMOSTRATIVO (Slide 21)
    # Creo delle fasce per analizzare la composizione della clientela.
    # Uso bins=[-1, 12, 48, 999] invece di [0, 12, 48, 999]: pd.cut per
    # default usa intervalli chiusi a destra (0, 12], quindi con lower bound
    # a 0 i clienti con tenure esattamente 0 (gli stessi 11 con TotalCharges
    # mancante) restavano fuori da ogni fascia e sparivano dal conteggio.
    labels_fasce = ['Nuovi (0-12)', 'Fedeli (12-48)', 'Storici (>48)']
    fasce_clienti = pd.cut(df['tenure'], bins=[-1, 12, 48, 999], labels=labels_fasce)

    print("    -> Step 2: Binning (Discretizzazione 'tenure')")
    print("       Distribuzione clienti per fascia:")
    print(fasce_clienti.value_counts().to_string(header=False))
    # Nota: uso questa colonna solo per la stampa sopra. 'tenure' normalizzato
    # (step 4) resta comunque la versione più precisa da dare in pasto al KNN.

    # 3. Encoding delle variabili categoriche.
    # get_dummies non calcola nessuna statistica aggregata sui dati (media,
    # quantili, min/max...): codifica ogni riga in base al solo valore della
    # sua categoria, quindi farlo prima dello split non causa data leakage.
    df = pd.get_dummies(df, drop_first=True)

    # Identificazione Target
    if 'Churn_Yes' in df.columns:
        y = df['Churn_Yes']
        X = df.drop(columns=['Churn_Yes'])
    else:
        # Fallback sicuro
        target = [c for c in df.columns if 'Churn' in c][0]
        y = df[target]
        X = df.drop(columns=[target])

    # 4. Split Stratificato — PRIMA di calcolare mediana, quantili IQR e
    # min/max per lo scaling. Se questi si calcolano su tutto il dataset e
    # solo dopo si fa lo split (come nella versione originale), il test set
    # ha già influenzato le statistiche usate per "pulire" il training set:
    # è data leakage, anche se qui l'effetto pratico è piccolo.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Valori mancanti in TotalCharges: mediana calcolata SOLO sul training set.
    mediana_train = X_train['TotalCharges'].median()
    X_train['TotalCharges'] = X_train['TotalCharges'].fillna(mediana_train)
    X_test['TotalCharges'] = X_test['TotalCharges'].fillna(mediana_train)

    # 5. GESTIONE OUTLIER (Boxplot & IQR - Slide 23)
    # Q1/Q3 calcolati solo sul training set, e la riga outlier si scarta solo
    # dal training: il test set deve restare come sarebbe in produzione, dove
    # non si possono scartare clienti veri solo perché "anomali".
    Q1 = X_train['TotalCharges'].quantile(0.25)
    Q3 = X_train['TotalCharges'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_MOLTIPLICATORE * IQR
    upper = Q3 + IQR_MOLTIPLICATORE * IQR

    n_prima = len(X_train)
    dentro_i_bound = (X_train['TotalCharges'] >= lower) & (X_train['TotalCharges'] <= upper)
    X_train, y_train = X_train[dentro_i_bound], y_train[dentro_i_bound]
    print(f"    -> Step 3: Pulizia Outlier (Metodo IQR) completata. "
          f"Righe rimosse dal training: {n_prima - len(X_train)}")

    # 6. NORMALIZZAZIONE (Min-Max - Slide 23)
    # Cruciale per il KNN: porta tutto su scala 0-1. Fit SOLO sul training,
    # poi applico la stessa trasformazione (transform, non fit_transform) al
    # test set, così il test non contribuisce a stabilire il min/max usato.
    scaler = MinMaxScaler()
    X_train[COLONNE_DA_SCALARE] = scaler.fit_transform(X_train[COLONNE_DA_SCALARE])
    X_test[COLONNE_DA_SCALARE] = scaler.transform(X_test[COLONNE_DA_SCALARE])
    print("    -> Step 4: Normalizzazione Min-Max applicata (fit solo sul training).")

    # 7. Scelta di k tramite 5-fold cross-validation sul training set, invece
    # di un valore fisso deciso a mano: così il valore scelto è giustificabile
    # e non è un altro parametro arbitrario che confonde il confronto col Baseline.
    print("    -> Step 5: Ricerca del k migliore (5-fold cross-validation)...")
    k_migliore, punteggio_migliore = None, -1.0
    for k in VALORI_K_DA_TESTARE:
        punteggio = cross_val_score(
            KNeighborsClassifier(n_neighbors=k), X_train, y_train, cv=5
        ).mean()
        if punteggio > punteggio_migliore:
            k_migliore, punteggio_migliore = k, punteggio
    print(f"       k scelto: {k_migliore} (accuratezza media in CV: {punteggio_migliore:.2%})")

    print(f"    -> Addestramento KNN Ottimizzato (k={k_migliore})...")
    model = KNeighborsClassifier(n_neighbors=k_migliore)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"    -> Finito. Accuratezza KDD: {acc:.2%}")
    print("\n    [Report Classificazione KDD]")
    print(classification_report(y_test, y_pred))

    return acc, cm
