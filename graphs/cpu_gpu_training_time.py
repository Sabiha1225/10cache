import matplotlib.pyplot as plt
import numpy as np

categories = ['OPT-6.7B', 'Bloom-7B', 'Falcon-7B']

Baseline2 = [None, None, None] 
Baseline1 = [4469.19, 9444.80, 27671.92]
Smart_Cache = [5078.11, 10789.78, 29119.76]
Smart_Cache_Pref = [3538.3, 7673.4, 24846.36]
Smart_Cache_Pref_Mem = [1758.34, 3435.88, 10581.84]
l2l = [10562.41, 18902.54, 55518.38]

data = [Baseline2, Baseline1, l2l, Smart_Cache, Smart_Cache_Pref, Smart_Cache_Pref_Mem]
labels = ['ZeRO-Offload', 'ZeRO-Infinity', 'L2L', 'SC', 'SC+P', 'SC+P+M']

fig, ax = plt.subplots(figsize=(4, 2.5))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['black', '#63993D', '#F8AE54', '#4394E5', "#58508d", "#003f5c"]
patterns = ["", "-", '.', "/", "\\", "+"]

for i, d in enumerate(data):
    if i == 0:  
        for j in range(len(categories)):
            ax.text(index[j] + (i * bar_width) + (i * bar_spacing), 0, 'X',
                   ha='center', va='bottom', fontsize=12, color='black',
                   fontweight='bold')
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width,
               label=labels[i], color=colors[i], edgecolor='white',
               hatch=patterns[i], zorder=3)

ax.text(index[0]+ 0.22, max(Baseline1) * 0.4, 'GPU OOM',
        ha='center', va='bottom', fontsize=8, color='red')

ax.set_ylabel('Training Time (s)')
ax.set_xticks(index + bar_width * 2.5)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='0.9')
ax.grid(axis='y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('training_time_cpu_gpu_6b_range.pdf')

plt.show()