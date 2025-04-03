import numpy as np
import pandas as pd
import seaborn as sn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Creazione modello della rete neurale
def create_model(input_dim):
    network = Sequential()
    network.add(Dense(30, input_dim=input_dim, activation="relu"))
    network.add(Dense(1, activation='sigmoid'))
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return network

# Funzione per pulire e trasformare i dati
def clean_data(data):
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 2})
    data['Dyslexia'] = data['Dyslexia'].map({'No': 0, 'Yes': 1})
    data['Nativelang'] = data['Nativelang'].map({'No': 0, 'Yes': 1})
    data['Otherlang'] = data['Otherlang'].map({'No': 0, 'Yes': 1})

# Caricamento dataset
data = pd.read_csv('/content/Dyslexia_dataset.csv', sep=';', encoding='utf-8')
clean_data(data)

# Selezione delle colonne di interesse
features = ['Gender', 'Nativelang', 'Otherlang', 'Age', 'Dyslexia']
for i in range(30):
    if (i in list(range(12)) + list(range(13, 17)) + [21, 22, 29]):
        for metric in ['Clicks', 'Hits', 'Misses', 'Score', 'Accuracy', 'Missrate']:
            features.append(f'{metric}{i+1}')

data = data[features]

# Separazione delle variabili dipendenti e indipendenti
y = data['Dyslexia']
X = data.drop(columns=['Dyslexia'])

# Bilanciamento con SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# Divisione in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=13)

# Cross-validation manuale con grafico per ogni fold
cv = KFold(n_splits=5, shuffle=True, random_state=13)

accuracies = []
f1_scores = []
precisions = []
recalls = []
all_y_true = []
all_y_pred = []

for i, (train_index, val_index) in enumerate(cv.split(X_train, y_train)):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = create_model(X_train_fold.shape[1])
    model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=64, verbose=0)

    predictions = model.predict(X_val_fold).flatten()
    predictions = np.round(predictions)

    # Metriche per ogni fold
    acc = accuracy_score(y_val_fold, predictions)
    f1 = f1_score(y_val_fold, predictions)
    prec = precision_score(y_val_fold, predictions)
    rec = recall_score(y_val_fold, predictions)

    accuracies.append(acc)
    f1_scores.append(f1)
    precisions.append(prec)
    recalls.append(rec)
    all_y_true.extend(y_val_fold)
    all_y_pred.extend(predictions)

    print(f'\n===== Fold {i+1} =====')
    print(f'Accuracy: {acc}')
    print('Classification Report:')
    print(classification_report(y_val_fold, predictions))

# Risultati aggregati
y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)
print('\n=== Risultati aggregati ===')
print('Classification Report:')
print(classification_report(y_true, y_pred))

# Visualizzazione grafico della performance per fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), accuracies, marker='o', label='Accuracy', linestyle='-', color='blue')
plt.plot(range(1, 6), f1_scores, marker='s', label='F1-score', linestyle='--', color='orange')
plt.plot(range(1, 6), precisions, marker='^', label='Precision', linestyle=':', color='green')
plt.plot(range(1, 6), recalls, marker='v', label='Recall', linestyle='-.', color='red')

plt.title('Cross-Validation Performance per Fold (Neural Network)')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(range(1, 6))
plt.grid(True)
plt.legend()
plt.show()

# Calcolo curva ROC e AUC
probs = model.predict(X_test).flatten()
from sklearn.metrics import roc_auc_score, roc_curve

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)

fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Curva Precision-Recall
from sklearn.metrics import average_precision_score, precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, probs)
average_precision = average_precision_score(y_test, probs)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall curve (AP={average_precision:.2f})')
plt.show()
