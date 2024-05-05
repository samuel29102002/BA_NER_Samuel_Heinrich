import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Aktualisierte Ergebnisse für jede Metrik
results = np.array([
    [0.6904274603824337, 0.6904274603824337, 0.6904274603824337, 0.9280743816151102],  # English
    [0.744105310888171, 0.744105310888171, 0.744105310888171, 0.9140735702544919],  # Spanish
    [0.7866303925378935, 0.7866303925378935, 0.7866303925378934, 0.839459219365943]   # Low-Resource
])

# Labels für die Achsen
metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
languages = ['English', 'Spanish', 'Low-Resource']

# Heatmap erstellen
plt.figure(figsize=(10, 6))
sns.heatmap(results, annot=True, fmt=".3f", cmap='Greens', xticklabels=metrics, yticklabels=languages)
plt.title('Heatmap der Modellevaluationsergebnisse')
plt.xlabel('Metriken')
plt.ylabel('Sprachen')

# Zeigt die Heatmap an
plt.show()
