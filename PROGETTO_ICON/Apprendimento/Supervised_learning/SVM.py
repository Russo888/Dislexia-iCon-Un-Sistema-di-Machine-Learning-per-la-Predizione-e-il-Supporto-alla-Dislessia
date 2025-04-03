import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_score, recall_score, precision_recall_curve, auc
)
from imblearn.over_sampling import SMOTE

# Funzione per la Precision-Recall Curve
def plot_precision_recall(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

# Funzione per Bar Chart di Varianza e Deviazione Standard
def plot_variance_std(acc, f1, prec, recall):
    metrics = ['Accuracy', 'F1-score', 'Precision', 'Recall']
    values = [np.var(acc), np.var(f1), np.var(prec), np.var(recall)]
    std_values = [np.std(acc), np.std(f1), np.std(prec), np.std(recall)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    ax[0].set_title('Varianza delle Metriche')
    ax[0].set_ylabel('Varianza')
    ax[0].grid(axis='y')

    ax[1].bar(metrics, std_values, color=['blue', 'orange', 'green', 'red'])
    ax[1].set_title('Deviazione Standard delle Metriche')
    ax[1].set_ylabel('Deviazione Standard')
    ax[1].grid(axis='y')

    plt.tight_layout()
    plt.show()

# Caricamento e preparazione dati (come definito in precedenza)
data = pd.read_csv("/content/Dyslexia_dataset.csv", sep=';', encoding='utf-8')
# Pulizia e selezione feature
# ...

# === Train/Test Split ===
svm_clf_split = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
svm_clf_split.fit(X_train_bal, y_train_bal)
y_pred_split = svm_clf_split.predict(X_test)
y_scores_split = svm_clf_split.decision_function(X_test)

plot_precision_recall(y_test, y_scores_split)

# === Cross-Validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
acc_list, f1_list, prec_list, recall_list = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train_cv, y_train_cv)

    svm_clf_cv = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    svm_clf_cv.fit(X_train_sm, y_train_sm)
    y_pred_cv = svm_clf_cv.predict(X_val_cv)
    y_scores_cv = svm_clf_cv.decision_function(X_val_cv)

    acc = accuracy_score(y_val_cv, y_pred_cv)
    f1 = f1_score(y_val_cv, y_pred_cv)
    prec = precision_score(y_val_cv, y_pred_cv)
    recall = recall_score(y_val_cv, y_pred_cv)

    acc_list.append(acc)
    f1_list.append(f1)
    prec_list.append(prec)
    recall_list.append(recall)

    print(f"\nFold {fold} - Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val_cv, y_pred_cv))

plot_variance_std(acc_list, f1_list, prec_list, recall_list)