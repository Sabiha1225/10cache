
import matplotlib.pyplot as plt
import numpy as np

categories = ['OPT-13B', 'Falcon-10B', 'Falcon-11B']

Baseline1 = [10.38, 3.31, 9.34]
Mine_Approach = [100.0, 99.64, 100.0]


data = [Baseline1, Mine_Approach]
labels = ['ZeRO-Infinity', 'SC+FP16+Opt-States']

fig, ax = plt.subplots(figsize=(4, 2))

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
plt.savefig('hit_rate_cpu_gpu_nvme_fp16.pdf')


plt.show()