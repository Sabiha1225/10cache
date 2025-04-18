import pandas as pd
import matplotlib.pyplot as plt

def plot_function(number):
    df1 = pd.read_csv('/content/drive/MyDrive/research/opt_125_M_trace_fx_graph/tensor_frequency_dict.csv', nrows=number)
    df2 = pd.read_csv('/content/drive/MyDrive/research/opt_125_M_trace_fx_graph/detailed_log_1.csv')

    df1['operation_name'] = df1['operation_name'].str.split(';')

    merged_df = pd.merge(df1.explode('operation_name'), df2, how='inner', on='operation_name')
    
    unique_tensors = merged_df['tensor_name'].unique()
    for i, old_name in enumerate(unique_tensors):
      print(f"old_name: {old_name} mapped_name: tensor{i+1}")
    tensor_mapping = {old_name: f'tensor{i+1}' for i, old_name in enumerate(unique_tensors)}

    merged_df['tensor_name'] = merged_df['tensor_name'].map(tensor_mapping)

    merged_df['cumulative_time(ms)'] = pd.to_numeric(merged_df['cumulative_time(ms)'])

    fig, ax = plt.subplots(figsize=(5,2))

    for tensor_name in merged_df['tensor_name'].unique():
        tensor_data = merged_df[merged_df['tensor_name'] == tensor_name]
        ax.scatter(tensor_data['cumulative_time(ms)'], [tensor_name]*len(tensor_data))

    ax.set_xlabel('Cumulative Time (ms)', fontsize=10)
    ax.set_ylabel('Tensor Name', fontsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    plt.savefig('tensor_timeline.pdf')

    plt.show()

plot_function(6)