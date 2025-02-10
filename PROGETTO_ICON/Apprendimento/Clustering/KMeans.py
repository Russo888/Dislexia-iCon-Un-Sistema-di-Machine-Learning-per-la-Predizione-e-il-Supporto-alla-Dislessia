import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score

# Caricamento del dataset
dataset = pd.read_csv("/content/Dyslexia_dataset.csv", sep=";", encoding="utf-8")

# Selezione delle feature di interesse
features = ['Gender', 'Nativelang', 'Otherlang', 'Age']
for i in range(30):
    if i in list(range(12)) + list(range(13, 17)) + [21, 22, 29]:
        for metric in ['Clicks', 'Hits', 'Misses', 'Score', 'Accuracy', 'Missrate']:
            features.append(f'{metric}{i+1}')

# Creazione del dataset filtrato
dataset = dataset[features]

# Identificazione delle colonne categoriche
categorical_features = ['Gender', 'Nativelang', 'Otherlang']

# One-hot encoding per le variabili categoriche
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = onehot_encoder.fit_transform(dataset[categorical_features])

# Standardizzazione delle feature numeriche
numerical_features = [col for col in dataset.columns if col not in categorical_features]
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(dataset[numerical_features])

# Creazione del dataset finale per il clustering
final_dataset = np.hstack((encoded_categorical, scaled_numerical))

# Determinazione del numero ottimale di cluster con il metodo del gomito
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(final_dataset)
    wcss.append(kmeans.inertia_)

# Grafico del metodo del gomito
plt.plot(range(1, 11), wcss, 'bx-')
plt.xlabel('Numero di cluster (K)')
plt.ylabel('WCSS')
plt.title('Metodo del gomito')
plt.show()

# Applicazione di K-Means con il numero di cluster ottimale (da scegliere in base al grafico)
optimal_k = 3  # Sostituiscilo con il valore corretto in base al grafico
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
clusters = kmeans.fit_predict(final_dataset)

# Aggiunta dei cluster al dataset originale
dataset['Cluster'] = clusters

# Valutazione con Silhouette Score
silhouette_avg = silhouette_score(final_dataset, clusters)
print(f'Silhouette Score: {silhouette_avg:.3f}')

# Salvataggio del dataset con cluster assegnati
dataset.to_csv('Dyslexia_Dataset_Clustered.csv', index=False)

print("Clustering completato e dataset salvato.")
