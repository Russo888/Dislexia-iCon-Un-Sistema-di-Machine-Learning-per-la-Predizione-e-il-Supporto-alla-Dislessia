import pandas as pd
import matplotlib.pyplot as plt

# Caricamento dei dati
data = pd.read_csv("Dyslexia_dataset", sep=";")

# Seleziona le colonne desiderate per le domande e Dyslexia
selected_columns = ["Score1", "Score2", "Score3", "Score4", "Score5", "Score6", "Score7",
                    "Score8", "Score9", "Score10", "Score11", "Score12", "Score13", "Score14",
                    "Score15", "Score16", "Score17", "Score18", "Score19", "Score20", "Score21",
                    "Score22", "Score23", "Score24", "Score25", "Score26", "Score27", "Score28",
                    "Score29", "Score30", "Score31", "Score32", "Dyslexia"]
# Filtra il dataset con le colonne selezionate
filtered_data = data[selected_columns]

# Calcola le medie delle risposte per i gruppi "Dyslexia"
mean_responses = filtered_data.groupby("Dyslexia").mean()

# Crea un grafico a barre per le differenze nelle medie delle risposte
plt.figure(figsize=(10, 6))
ax = mean_responses.T.plot(kind="bar", color=["skyblue", "orange"], figsize=(15, 6))
plt.title("Differenze nelle risposte tra gruppo Dyslexia 'yes' o 'no'")
plt.xlabel("Score")
plt.ylabel("Media dello Score")
plt.xticks(rotation=45, ha="right")

# Stampa i valori sulle colonne
for container in ax.containers:
    ax.bar_label(container, fmt='%2.2f', label_type='edge', color='black')

plt.legend(["Non Dislessico", "Dislessico"])
plt.show()