import os
import ast
import wfdb

import numpy as np
import pandas as pd
import wandb

from argparse import ArgumentParser
from pathlib import Path
from functools import partial




def parse_args():
    
    parser = ArgumentParser()
    parser.add_argument('--raw_data_dir', type=str, default='./data/ptbxl')
    parser.add_argument('--output_dir', type=str, default='./data/ptbxl_split')
    parser.add_argument('--sampling_rate', type=int, default=100)
    parser.add_argument('--wandb_project', type=str, default='ecg_benchmarking_lit')
    parser.add_argument('--dataset_name', type=str, default='ptbxl_split')
    
    args = parser.parse_args()
    
    return Path(args.raw_data_dir), Path(args.output_dir), args.sampling_rate, args.wandb_project, args.dataset_name


def load_and_process_ecg_df(path_to_data):
    ECG_df = pd.read_csv(path_to_data / 'ptbxl_database.csv', index_col='ecg_id')
    ECG_df.scp_codes = ECG_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    ECG_df.patient_id = ECG_df.patient_id.astype(int)
    ECG_df.nurse = ECG_df.nurse.astype('Int64')
    ECG_df.site = ECG_df.site.astype('Int64')
    ECG_df.validated_by = ECG_df.validated_by.astype('Int64')
    
    return ECG_df


def diagnostic_class(scp, SCP_df):
    res = set()
    for k in scp.keys():
        if k in SCP_df.index:
            res.add(SCP_df.loc[k].diagnostic_class)
    return list(res)
                    



def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def save_files_according_to_split(df, raw_data_path, prefix="train", output_dir="./data/ptbxl_split", sampling_rate=100):
    encoded_df = df.loc[:, ['patient_id'] + unique_classes.tolist()]
    metadata = df.drop(columns=unique_classes.tolist())
    data = load_raw_data(df, sampling_rate, raw_data_path)

    np.save(os.path.join(output_dir, f"{prefix}_data.npy"), data)
    metadata.to_csv(os.path.join(output_dir, f"{prefix}_metadata.csv"))
    encoded_df.to_csv(os.path.join(output_dir, f"{prefix}_labels.csv"))

    return encoded_df, metadata, data


def create_binary_encoded(labels, label_mapping):
    res = np.zeros(5)
    for l in labels:
        res[label_mapping[l]] = 1
    return res

def get_filtered_scp_df(raw_data_path):
    SCP_df = pd.read_csv(raw_data_path / 'scp_statements.csv', index_col=0)
    SCP_df = SCP_df[SCP_df.diagnostic == 1]
    return SCP_df



def get_unique_classes(SCP_df):
    unique_classes = SCP_df.diagnostic_class.unique()
    return unique_classes

if __name__ == "__main__":
    
    input_dir, output_dir, sampling_rate, wandb_project, dataset_name = parse_args()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ECG_df = load_and_process_ecg_df(input_dir)
    SCP_df = get_filtered_scp_df(input_dir)
    unique_classes = get_unique_classes(SCP_df)

    label_mapping = {c: i for i, c in enumerate(unique_classes)}

    create_binary_encoded = partial(create_binary_encoded, label_mapping=label_mapping)
    
    diagnostic_class = partial(diagnostic_class, SCP_df=SCP_df)
    ECG_df['scp_classes'] = ECG_df.scp_codes.apply(diagnostic_class)

    ECG_df.loc[:, unique_classes.tolist()] =  np.vstack(ECG_df['scp_classes'].apply(create_binary_encoded))

    ECG_df_train = ECG_df[ECG_df.strat_fold < 8]
    ECG_df_test = ECG_df[ECG_df.strat_fold == 9]
    ECG_df_val = ECG_df[ECG_df.strat_fold == 8]
    
    encoded_df_train, metadata_train, data_train = save_files_according_to_split(ECG_df_train, input_dir, prefix='train', sampling_rate=sampling_rate)
    encoded_df_val, metadata_val, data_val = save_files_according_to_split(ECG_df_val, input_dir, prefix='val', sampling_rate=sampling_rate)
    encoded_df_test, metadata_test, data_test = save_files_according_to_split(ECG_df_test, input_dir, prefix='test', sampling_rate=sampling_rate)
    
    wandb.init(project=wandb_project)
    artifact = wandb.Artifact(name=dataset_name, type="dataset")
    artifact.add_dir(local_path=output_dir)
    wandb.log_artifact(artifact)