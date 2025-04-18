import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(4, 3))

file_path1 = '/content/drive/MyDrive/research/cpu-gpu-nvme_measuring_files/wait_time_cpu_gpu_nvme_param16_opt_13_b_mine.csv'
df1 = pd.read_csv(file_path1, usecols=[0, 1], skiprows=range(1, 1291), nrows=1290)
df1 = df1.sort_values(by="Wait_Time_loading(ms)")
cdf1 = df1["Wait_Time_loading(ms)"].cumsum() / df1["Wait_Time_loading(ms)"].sum()

below_threshold_count1 = (df1["Wait_Time_loading(ms)"] < 0.03).sum()
percentage1 = (below_threshold_count1 / len(df1)) * 100

file_path2 = '/content/drive/MyDrive/research/cpu-gpu-nvme_measuring_files/wait_time_cpu_gpu_nvme_param16_opt_13_b_deepspeed_bs1.csv'
df2 = pd.read_csv(file_path2, usecols=[0, 1])
df2 = df2.sort_values(by="Wait_Time_loading(ms)")
cdf2 = df2["Wait_Time_loading(ms)"].cumsum() / df2["Wait_Time_loading(ms)"].sum()


below_threshold_count2 = (df2["Wait_Time_loading(ms)"] < 0.03).sum()
percentage2 = (below_threshold_count2 / len(df2)) * 100


plt.plot(df1["Wait_Time_loading(ms)"], cdf1,
         marker='.', markersize=6, linestyle='-', linewidth=1.5,
         color='#00b4d8', label="SC+FP16+Opt-States")
plt.plot(df2["Wait_Time_loading(ms)"], cdf2,
         marker='.', markersize=6, linestyle='-', linewidth=1.5,
         color='#e56b6f', label="ZeRO-Infinity")

plt.axvline(x=0.03, color='gray', linestyle='--', linewidth=1, label='Threshold: 0.03 ms')

plt.text(0.0003, 0.85, f'{percentage1:.1f}% below 0.03 ms',
         color='#00b4d8', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
plt.text(0.0003, 0.55, f'{percentage2:.1f}% below 0.03 ms',
         color='#e56b6f', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.xlabel("Wait Time for CPU to GPU Swap-In (ms)")
plt.ylabel("Cumulative Probability")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.xscale('log')
plt.xlim(0.0001, 20)  
plt.ylim(-0.05, 1.05)  

plt.tight_layout()
plt.savefig('wait_time_cpu_gpu_nvme_param16_opt_13_b_both.pdf')
plt.show()