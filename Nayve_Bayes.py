import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB # arba BernoulliNB
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, auc, roc_auc_score)
# import sys

# Funkcija, kuri atlieka visą procesą konkrečiam failui
def process_naive_bayes(file_path, title_description):
    print(f"\n{'='*60}")
    print(f"Ruošiamas failas: {file_path} ({title_description})")
    print(f"{'='*60}")

    # 1. Duomenų įkėlimas
    try:
        df = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        print(f"KLAIDA: Failas '{file_path}' nerastas.")
        return

    df_clean = df.dropna()
    
    # Atrankos užtikrinimas (1000 eilučių)
    if len(df_clean) >= 1000:
        df_clean = df_clean.sample(n=1000, random_state=42)
        print("Pastaba: Atrinkta 1000 atsitiktinių eilučių.")
    else:
        print(f"Duomenų mažiau nei 1000 ({len(df_clean)}).")

    # Atskiriame X ir y
    y = df_clean['label']
    X = df_clean.drop('label', axis=1)

    # 2. Duomenų padalinimas
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=150, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=150, stratify=y_temp, random_state=42
    )

    print(f"Duomenų padalinimas: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. Normavimas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Modelio apmokymas
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)

    # 5. Prognozavimas ir Metrikos (Test aibei)
    y_pred = nb_model.predict(X_test_scaled)
    
    # Svarbu: ROC kreivėms reikia tikimybių (probabilities)
    y_prob = nb_model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # AUC skaičiavimas
    classes = np.unique(y)
    n_classes = len(classes)
    
    # Binarizuojame y_test, kad galėtume skaičiuoti AUC kiekvienai klasei
    y_test_bin = label_binarize(y_test, classes=classes)
    
    roc_auc_val = roc_auc_score(y_test, y_prob[:, 1])

    print(f"\n--- REZULTATAI ---")
    print(f"- Accuracy:  {acc:.4f}")
    print(f"- Precision: {prec:.4f}")
    print(f"- Recall:    {rec:.4f}")
    print(f"- F1:  {f1:.4f}")
    print(f"- AUC: {roc_auc_val:.4f}")

    # 6. Vizualizacija: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nSumaišymo matrica:")
    print(cm)

    # 7. Vizualizacija: ROC Kreivės
    print(f"Generuojamos ROC kreivės")
    plt.figure(figsize=(10, 8))
    output_plot = f"{file_path.split('.')}_ROC.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    
    # Iteruojame per kiekvieną klasę ir braižome jai kreivę
    for i in range(n_classes):
        class_label = classes[i]
        
        y_test_binary_class = (y_test == class_label).astype(int)
        
        y_prob_class = y_prob[:, i]
        
        fpr, tpr, _ = roc_curve(y_test_binary_class, y_prob_class)
        roc_auc_class = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'Klasė {class_label} (AUC = {roc_auc_class:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Įstrižainė (atsitiktinis spėjimas)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Klaidingai teigiamų dažnis')
    plt.ylabel('Jautrumas')
    plt.title(f'ROC Kreivės: {title_description}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. Vizualizacija: Sprendimų ribos
    if X.shape[1] == 2:
        print(f"Generuojamos sprendimų ribos")
        X_set, y_set = X_test_scaled, y_test.values
        X1, X2 = np.meshgrid(
            np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.05),
            np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.05)
        )
        Z = nb_model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X1, X2, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue', 'gray')))
        classes_unique = np.unique(y_set)
        colors = ['red', 'green', 'blue', 'gray']
        for i, j in enumerate(classes_unique):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(colors)(i), label=f'Klasė {j}', edgecolor='black', s=40)
        plt.title(f'Sprendimų ribos ({title_description})')
        plt.xlabel('TSNE 1')
        plt.ylabel('TSNE 2')
        plt.legend()
        plt.show()
        output_plot = f"{file_path.split('.')}_vizualizacija.png"
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')

# --- Vykdymas ---
process_naive_bayes('EKG_1000_originalus.csv', 'Originalūs EKG duomenys')
process_naive_bayes('EKG_1000_5.csv', 'Suspausti EKG duomenys')
process_naive_bayes('EKG_1000_tsne.csv', 't-SNE transformuoti duomenys')