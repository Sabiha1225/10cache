
import matplotlib.pyplot as plt
import numpy as np

groups = ['OPT-125M', 'OPT-350M', 'OPT-1.3B']
y0 = np.array([0.23, 0.37, 0.48])
y1 = np.array([0.25, 0.7, 2.6])
y2 = np.array([0.25, 0.7, 2.6])
y3 = np.array([0.5, 1.4, 5.2])
y4 = np.array([0.5, 1.4, 5.2])
y5 = np.array([0.5, 1.4, 5.2])
y6 = np.array([0.5, 1.4, 5.2])

fig, ax = plt.subplots(figsize=(4,2.5) )

ax.bar(groups, y0, label="Activation & Temp. Buffer", color="#ea801c", width=0.3)
ax.bar(groups, y1, bottom=y0, label="FP16 Param", color="#54bebe", width=0.3)
ax.bar(groups, y2, bottom=y0 + y1, label="FP16 Param Grad", color="#76c8c8", width=0.3)
ax.bar(groups, y3, bottom=y0 + y1 + y2, label="FP32 Param", color="#1a80bb", width=0.3)
ax.bar(groups, y5, bottom=y0 + y1 + y2 + y4, label="FP32 Momentum", color="#384860", width=0.3)
ax.bar(groups, y6, bottom=y0 + y1 + y2 + y4 + y5, label="FP32 Variance", color="#97a6c4", width=0.3)

ax.legend()
ax.set_ylabel('Memory (GB)')


plt.tight_layout()
plt.savefig('/content/sample_data/model_memory_requirements.pdf')

plt.show()