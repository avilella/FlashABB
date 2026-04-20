from matplotlib import pyplot as plt
import numpy as np

names = ['AFM', 'ABB2', 'ESM2', 'PDB\nloading', 'AbLang2', 'FlashABB']
times = [500, 5, 0.082, 0.017, 0.009, 0.0045]
colours = ['lightblue', 'lightblue', 'lightsalmon', 'lightgreen', 'lightsalmon', 'lightblue']

# Create figure with explicit size
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars with edgecolor
bars = ax.bar(names, times, color=colours, edgecolor='black', linewidth=1)

ax.set_yscale('log')
ax.set_ylabel('Time per Ab (s)', fontsize=14)
ax.set_title('Inference Speed Comparison', fontsize=16)

# Add value labels on bars
for bar, time in zip(bars, times):
    height = bar.get_height()
    if time >= 1:
        label = f'{time:.0f}s'
    elif time >= 0.01:
        label = f'{time*1000:.0f}ms'
    else:
        label = f'{time*1000:.1f}ms'
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,
            label, ha='center', va='bottom', fontsize=10)

# Style x-axis labels
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=11)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='lightblue', edgecolor='black', label='Structure prediction'),
    Patch(facecolor='lightsalmon', edgecolor='black', label='Sequence embedding'),
    Patch(facecolor='lightgreen', edgecolor='black', label='Data loading'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

plt.tight_layout()
plt.savefig('ab-speeds.pdf', dpi=150)
plt.savefig('ab-speeds.png', dpi=150)
print("Saved plots to ab-speeds.pdf and ab-speeds.png")
plt.show()
