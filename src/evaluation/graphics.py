import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the results for each metric
english_results = [0.7217535058716223, 0.8024973062052355, 0.7599867943215584, 0.9280743816151102]
spanish_results = [0.5695064088945542, 0.6751175285426461, 0.6178312142280952, 0.9140735702544919]
low_resource_results = [0.18924617817606748, 0.21111437812408115, 0.19958304378040306, 0.839459219365943]


true_positives_results = [47228, 52513, 28336]  # English, Spanish, Low-Resource
false_positives_results = [21176, 18059, 7686]
false_negatives_results = [21176, 18059, 7686]

error_analysis_results = np.array([
    [47228, 21176, 21176],  # True Positives, False Positives, False Negatives for English
    [52513, 18059, 18059],  # Spanish
    [28336, 7686, 7686]     # Low-Resource
])

# Colors for each bar
colors = ['deepskyblue', 'salmon', 'mediumseagreen']

# Set up individual figures for each metric
fig1, ax1 = plt.subplots(figsize=(6, 4))
fig2, ax2 = plt.subplots(figsize=(6, 4))
fig3, ax3 = plt.subplots(figsize=(6, 4))
fig4, ax4 = plt.subplots(figsize=(6, 4))

# Define the bar positions
bar_positions = np.arange(3)

# Plot Precision
ax1.bar(bar_positions, [english_results[0], spanish_results[0], low_resource_results[0]], color=colors)
ax1.set_xticks(bar_positions)
ax1.set_xticklabels(['English', 'Spanish', 'Low-Resource'])
ax1.set_title('Precision')
ax1.set_ylim(0, 1)

# Plot Recall
ax2.bar(bar_positions, [english_results[1], spanish_results[1], low_resource_results[1]], color=colors)
ax2.set_xticks(bar_positions)
ax2.set_xticklabels(['English', 'Spanish', 'Low-Resource'])
ax2.set_title('Recall')
ax2.set_ylim(0, 1)

# Plot F1-Score
ax3.bar(bar_positions, [english_results[2], spanish_results[2], low_resource_results[2]], color=colors)
ax3.set_xticks(bar_positions)
ax3.set_xticklabels(['English', 'Spanish', 'Low-Resource'])
ax3.set_title('F1-Score')
ax3.set_ylim(0, 1)

# Plot Accuracy
ax4.bar(bar_positions, [english_results[3], spanish_results[3], low_resource_results[3]], color=colors)
ax4.set_xticks(bar_positions)
ax4.set_xticklabels(['English', 'Spanish', 'Low-Resource'])
ax4.set_title('Accuracy')
ax4.set_ylim(0, 1)

# Save figures to the output directory
fig1.savefig('./data/precision_plot.png')
fig2.savefig('./data/recall_plot.png')
fig3.savefig('./data/f1score_plot.png')
fig4.savefig('./data/accuracy_plot.png')

# Error Analysis Results Plot
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.bar(bar_positions - 0.2, true_positives_results, width=0.2, color='deepskyblue', label='True Positives')
ax5.bar(bar_positions, false_positives_results, width=0.2, color='salmon', label='False Positives')
ax5.bar(bar_positions + 0.2, false_negatives_results, width=0.2, color='mediumseagreen', label='False Negatives')
ax5.set_xticks(bar_positions)
ax5.set_xticklabels(['English', 'Spanish', 'Low-Resource'])
ax5.set_title('Error Analysis Results')
ax5.set_ylabel('Counts')
ax5.legend()

# Save the Error Analysis Results figure
fig5.savefig('./data/error_analysis_results.png')

fig6, ax6 = plt.subplots(figsize=(8, 4))
sns.heatmap(error_analysis_results, annot=True, fmt="d", cmap='Greens', cbar=True, xticklabels=['True Positives', 'False Positives', 'False Negatives'], yticklabels=['English', 'Spanish', 'Low-Resource'])
ax6.set_title('Heatmap der Fehleranalyseergebnisse')
fig6.savefig('./data/error_analysis_heatmap.png')

# Close the plots to avoid displaying them in this notebook output
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)
plt.close(fig5)
plt.close(fig6)
