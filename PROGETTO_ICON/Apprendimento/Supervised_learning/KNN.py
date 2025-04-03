import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

# Pulizia dati
def clean_data(data):
    for col in data.columns:
        data[col] = data[col].astype('string')
        data[col] = data[col].astype('float', errors='ignore')
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})
    data['Dyslexia'] = data['Dyslexia'].map({'No': 0, 'Yes': 1})
    data['Nativelang'] = data['Nativelang'].map({'No': 0, 'Yes': 1})
    data['Otherlang'] = data['Otherlang'].map({'No': 0, 'Yes': 1})

# Caricamento dati
data = pd.read_csv("/content/Dyslexia_dataset.csv", sep=';', encoding='utf-8')
clean_data(data)

# Selezione feature
features = ['Gender', 'Nativelang', 'Otherlang', 'Age', 'Dyslexia']
for i in range(30):
    if i in list(range(12)) + list(range(13, 17)) + [21, 22, 29]:
        for metric in ['Clicks', 'Hits', 'Misses', 'Score', 'Accuracy', 'Missrate']:
            features.append(f'{metric}{i+1}')
data = data[features]

# Variabili indipendenti e target
y = data['Dyslexia']
X = data.drop(columns=['Dyslexia'])

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list, prec_list, recall_list = [], [], [], []
all_predictions, all_true = [], []

# Test su vari valori di K
error_rates = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(X, y)

    cv_scores = cross_val_score(knn, X_sm, y_sm, cv=cv)
    error_rates.append(1 - np.mean(cv_scores))

# Grafico dell'errore medio rispetto a K
plt.figure(figsize=(8, 5))
plt.plot(range(1, 20), error_rates, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.grid(True)
plt.show()

# Scelta di K ottimale
optimal_k = np.argmin(error_rates) + 1

# Addestramento con cross-validation
for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    knn_clf = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_clf.fit(X_train_sm, y_train_sm)
    y_pred = knn_clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    acc_list.append(acc)
    f1_list.append(f1)
    prec_list.append(prec)
    recall_list.append(recall)

    all_predictions.extend(y_pred)
    all_true.extend(y_val)

    # Stampa metriche per ogni fold
    print(f"\n===== Fold {fold} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {recall:.4f}")

# Grafico delle metriche di cross-validation
folds = list(range(1, len(acc_list) + 1))
plt.figure(figsize=(10, 6))
plt.plot(folds, acc_list, label='Accuracy', marker='o', linestyle='-', color='royalblue')
plt.plot(folds, f1_list, label='F1-score', marker='s', linestyle='--', color='orange')
plt.plot(folds, prec_list, label='Precision', marker='^', linestyle=':', color='green')
plt.plot(folds, recall_list, label='Recall', marker='v', linestyle='-.', color='red')
plt.title('Cross-Validation Performance per Fold (KNN)')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(folds)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# ROC Curve
y_probs = knn_clf.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_probs)
auc_score = roc_auc_score(y_val, y_probs)

plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc_score:.3f})')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, y_probs)
avg_precision = average_precision_score(y_val, y_probs)

plt.figure(figsize=(8, 5))
plt.step(recall, precision, where='post', color='b', alpha=0.6)
plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AP = {avg_precision:.3f})')
plt.show()

# Grafico varianza e deviazione standard
data_variance = {'Variance': np.var(acc_list), 'Standard Deviation': np.std(acc_list)}
names = list(data_variance.keys())
values = list(data_variance.values())

plt.figure(figsize=(6, 3))
plt.bar(names, values, color=['blue', 'orange'])
plt.title('Variance and Standard Deviation of Cross-Validation Scores')
plt.show()
