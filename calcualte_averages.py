import pandas as pd
import os
import numpy as np
from collections import defaultdict


base_paths = ['./train_output','./train_output_ep','./train_output_ep_es1','./train_output_ep_es6','./train_output_es1']



for base_path in base_paths:
    accuracy_mean = []
    accuracy_std = []
    orientation_error_mean = []
    orientation_error_std = []
    point_error_mean = []
    point_error_std = []
    categories = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    for category in categories:
        print(category)
        time = sorted(os.listdir(os.path.join(base_path,category,'fine_tuning')))[-1]
        final_path = os.path.join(base_path,category,'fine_tuning',time)
        splits = sorted([d for d in os.listdir(final_path) if os.path.isdir(os.path.join(final_path, d))])
        assert len(splits) == 3, f'{splits}'
        metrics = []
        for split in splits:
            csv_file = os.path.join(final_path,split,'metrics.csv')
            df = pd.read_csv(csv_file, header=None)
            row = df.iloc[0].to_numpy()
            metrics.append(row)

        metrics = np.array(metrics)
        mean = metrics.mean(axis=0)
        std = metrics.std(axis=0)
        df = pd.DataFrame([mean,std])
        output_file_path = os.path.join(final_path,'metrics.csv')
        df.to_csv(output_file_path, index=False, header=False)

        accuracy_mean.append(mean[-3])
        accuracy_std.append(std[-3])
        orientation_error_mean.append(mean[-2])
        orientation_error_std.append(std[-2])
        point_error_mean.append(mean[-1])
        point_error_std.append(std[-1])

    accuracy_mean = np.array(accuracy_mean).reshape(len(accuracy_mean),1)
    accuracy_std = np.array(accuracy_std).reshape(len(accuracy_std),1)
    orientation_error_mean = np.array(orientation_error_mean).reshape(len(orientation_error_mean),1)
    orientation_error_std = np.array(orientation_error_std).reshape(len(orientation_error_std),1)
    point_error_mean = np.array(point_error_mean).reshape(len(point_error_mean),1)
    point_error_std = np.array(point_error_std).reshape(len(point_error_std),1)

    accuracy_df = pd.DataFrame(np.concatenate((accuracy_mean,accuracy_std)))
    output_file_path = os.path.join(base_path,'accuracy_metrics.csv')
    accuracy_df.to_csv(output_file_path,float_format='%.2f', index=False, header=False)

    orientation_df = pd.DataFrame(np.concatenate((orientation_error_mean,orientation_error_std)))
    output_file_path = os.path.join(base_path,'orientation_metrics.csv')
    orientation_df.to_csv(output_file_path,float_format='%.2f', index=False, header=False)

    point_df = pd.DataFrame(np.concatenate((point_error_mean,point_error_std)))
    output_file_path = os.path.join(base_path,'point_metrics.csv')
    point_df.to_csv(output_file_path,float_format='%.2f', index=False, header=False)





    




