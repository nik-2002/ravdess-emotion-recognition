
import matplotlib.pyplot as plt
import pandas as pd

# Data
models = ['SVM (Baseline)', 'Random Forest', 'AutoML (MLP)', 'CNN (Mel Spec)']
accuracies = [22.69, 32.87, 62.76, 62.50]
colors = ['#bdc3c7', '#95a5a6', '#3498db', '#e74c3c'] # Grey, Grey, Blue, Red

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=colors)

# Add labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Model Performance Comparison: From Baseline to Deep Learning', fontsize=16)
plt.ylim(0, 80)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save
plt.tight_layout()
plt.savefig('../output/final_model_comparison.png', dpi=300)
print("Saved comparison chart to ../output/final_model_comparison.png")
