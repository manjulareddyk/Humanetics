import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def step_average(sample_array_a,start,step):
    back_sum=sample_array_a[(start-step):start].sum()
    front_sum=sample_array_a[start:(start+step)].sum()
    avg=((back_sum+front_sum)/(step*2))
    return avg

def average_replace_values(sample_array_d,start_index1,start_index2,step):
    range_step=start_index2-start_index1
    final_array=sample_array_d.copy()
    for values in range(0,range_step+1):
        avg=step_average(final_array,(values+start_index1),step)
        sample_array_d[values+start_index1]=avg
    return sample_array_d

def main():
    file_csv=input('please give csv file')
    df=pd.read_csv(file_csv)
    df2=df
    skip_coloums=['Time__UTC_']
    df2=df2.drop(columns=skip_coloums)
    featurtes_list = df2.columns.to_list()
    array_df = np.zeros(shape=(len(df2.columns.to_list()), 360, 4000), dtype=float)
    print(f"array shape:{array_df.shape}")

    class_index_1 = df2[df2['class_labels'] == 1].index[0]
    print(class_index_1)
    class_1_samples = 10
    sample_files = 36
    total_sample = sample_files * class_1_samples
    step_index = class_index_1
    step=100
    for index_f, features in enumerate(featurtes_list):
        start_sample = 0
        step_sample = 2000
        for samples in range(0, total_sample, class_1_samples):
            sample_array = np.zeros(4000)
            for index_1_sample in range(0, class_1_samples):
                sample_array[0:2000] = df2[features][start_sample:start_sample + 2000]
                sample_array[2000:] = df2[features][step_index:step_index + 2000]
                avg=average_replace_values(sample_array, 1899, 2299,step)#average step
                array_df[index_f][samples + index_1_sample] = avg
                step_index = step_index + 2000
            start_sample = start_sample + 2000
            step_index = class_index_1
    for features_graphs in range(0, (len(featurtes_list) - 7), 8):
        # Assuming array_df[17][0], array_df[16][0], and array_df[15][0] are your data
        subplots = [array_df[features_graphs + i][0] for i in range(8)]

        # Create some fake data (replace with your actual data)
        xs = [np.arange(len(subplot)) for subplot in subplots]

        # Create subplots
        fig, axs = plt.subplots(8, 1, figsize=(10, 10))
        #fig.suptitle('Belt tension ')

        for i, ax in enumerate(axs):
            ax.plot(xs[i][:2000], subplots[i][:2000], '.-', label="Class zero", color='green')
            ax.plot(xs[i][2000:], subplots[i][2000:], '.-', label="Class One", color='orange')
            ax.set_xlabel('Time in Msecs')
            ax.set_title(featurtes_list[features_graphs + i])
            ax.legend()

        # Adjust layout to prevent overlap
        plt.subplots_adjust(hspace=0.5)

        # Automatically adjust layout
        plt.tight_layout()
        path=os.getcwd()
        # Save the figure as a PNG file
        dir = f'{path}/graphs_{step}/subplots/'
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        plt.savefig(f'{dir}{features_graphs}multiple_subplots.png')
        plt.close()

if __name__=="__main__":
    main()

