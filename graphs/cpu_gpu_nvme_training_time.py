
import matplotlib.pyplot as plt
import numpy as np

categories = ['OPT-13B', 'Falcon-10B', 'Falcon-11B']

Baseline1 = [179400.41, 194424.66, 205330.16]
Mine_Approach_with_FP16 = [157649.30, 167100.59, 182709.06]
Mine_Approach_with_Both = [145080.30, 150285.71, 155424.31]


data = [Baseline1, Mine_Approach_with_FP16, Mine_Approach_with_Both]
labels = ['ZeRO-Infinity', 'SC+FP16', 'SC+FP16+Opt-States']

fig, ax = plt.subplots(figsize=(4, 2.5))

bar_width = 0.15
bar_spacing = 0.009
index = np.arange(len(categories))


colors = ['#63993D', '#4394E5', "#c86558"]
patterns = ["-", "/", "\\"]

for i, d in enumerate(data):
    if patterns[i] == "":
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color='white', edgecolor=colors[i], hatch="/", zorder=3)
    else:
        ax.bar(index + (i * bar_width) + (i * bar_spacing), d, bar_width, label=labels[i], color=colors[i], edgecolor='white', hatch=patterns[i], zorder=3)

ax.set_ylabel('Training Time (s)')
ax.set_xticks(index + bar_width * 1)
ax.set_xticklabels(categories)
ax.legend(frameon=True, edgecolor='0.9')
ax.grid(axis = 'y', zorder=0, color='#C0C0C0')

plt.tight_layout()
plt.savefig('training_time_cpu_gpu_nvme_10b_range.pdf')

plt.show()