import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

# Funzione per la pulizia dei dati
def clean_data(data):
    for col in data.columns:
        data[col] = data[col].astype('string')
        data[col] = data[col].astype('float', errors='ignore')
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})
    data['Dyslexia'] = data['Dyslexia'].map({'No': 0, 'Yes': 1})
    data['Nativelang'] = data['Nativelang'].map({'No': 0, 'Yes': 1})
    data['Otherlang'] = data['Otherlang'].map({'No': 0, 'Yes': 1})

# Funzione per visualizzare le metriche della cross-validation
def plot_cv_metrics(acc, f1, prec, recall):
    folds = list(range(1, len(acc) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(folds, acc, label='Accuracy', marker='o', linestyle='-', color='royalblue')
    plt.plot(folds, f1, label='F1-score', marker='s', linestyle='--', color='orange')
    plt.plot(folds, prec, label='Precision', marker='^', linestyle=':', color='green')
    plt.plot(folds, recall, label='Recall', marker='v', linestyle='-.', color='red')
    plt.title('Cross-Validation Performance per Fold (Random Forest)')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.xticks(folds)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === INIZIO SCRIPT ===

data = pd.read_csv("/content/Dyslexia_dataset.csv", sep=';', encoding='utf-8')
clean_data(data)

# Selezione delle feature
features = ['Gender', 'Nativelang', 'Otherlang', 'Age', 'Dyslexia']
for i in range(30):
    if i in list(range(12)) + list(range(13, 17)) + [21, 22, 29]:
        for metric in ['Clicks', 'Hits', 'Misses', 'Score', 'Accuracy', 'Missrate']:
            features.append(f'{metric}{i+1}')
data = data[features]

# Separazione tra variabili indipendenti (X) e target (y)
y = data['Dyslexia']
X = data.drop(columns=['Dyslexia'])

# Creazione del metodo di validazione incrociata
cv = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list, prec_list, recall_list = [], [], [], []
all_predictions, all_true = [], []

for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
    print(f"\n===== Fold {fold} =====")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Bilanciamento dei dati con SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    # Creazione e addestramento del Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        class_weight={0:1, 1:3},
        random_state=42
    )
    rf_clf.fit(X_train_sm, y_train_sm)
    y_pred = rf_clf.predict(X_val)

    # Calcolo delle metriche per il fold corrente
    acc_list.append(accuracy_score(y_val, y_pred))
    f1_list.append(f1_score(y_val, y_pred))
    prec_list.append(precision_score(y_val, y_pred))
    recall_list.append(recall_score(y_val, y_pred))

    all_predictions.extend(y_pred)
    all_true.extend(y_val)

    # Output delle metriche
    print("Accuracy:", acc_list[-1])
    print("Classification Report:\n", classification_report(y_val, y_pred))

# Report complessivo aggregato
print("\n=== Risultati aggregati ===")
print("Classification Report:\n", classification_report(all_true, all_predictions))

# Grafico delle metriche della cross-validation
plot_cv_metrics(acc_list, f1_list, prec_list, recall_list)

# Curva ROC
y_probs = rf_clf.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_probs)
plt.figure()
plt.plot(fpr, tpr, linestyle='-', color='blue', label='ROC curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_probs)
plt.figure()
plt.plot(recall, precision, marker='.', color='green', label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
