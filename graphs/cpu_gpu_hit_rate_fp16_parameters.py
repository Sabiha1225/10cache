
import matplotlib.pyplot as plt
import numpy as np

categories = ['OPT-6.7B', 'Bloom-7B', 'Falcon-7B']

Baseline1 = [3.68, 17.21, 1.03]
Mine_Approach = [100.0, 83.87, 89.18]

data = [Baseline1, Mine_Approach]
labels = ['ZeRO-Infinity', 'SC+P+M']

fig, ax = plt.subplots(figsize=(4, 2.5))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))

colors = ['#63993D', '#4394E5']
patterns = ["-", "/"]

for i, d in enumerate(data):
    if patterns[i] == "":
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color='white', edgecolor=colors[i], hatch="/", zorder=3)
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color=colors[i], edgecolor='white', hatch=patterns[i], zorder=3)

ax.set_ylabel('Hit Rate')
ax.set_xticks(index + bar_width * 1)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='0.9')
ax.grid(axis = 'y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('hit_rate_cpu_gpu.pdf')

plt.show()