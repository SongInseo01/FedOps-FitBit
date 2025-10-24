import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""
# Pytorch version

# MNIST
# def load_partition(dataset, validation_split, batch_size):
#     """
#     The variables train_loader, val_loader, and test_loader must be returned fixedly.
#     """
#     now = datetime.now()
#     now_str = now.strftime('%Y-%m-%d %H:%M:%S')
#     fl_task = {"dataset": dataset, "start_execution_time": now_str}
#     fl_task_json = json.dumps(fl_task)
#     logging.info(f'FL_Task - {fl_task_json}')

#     # MNIST Data Preprocessing
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
#     ])

#     # Download MNIST Dataset
#     full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

#     # Splitting the full dataset into train, validation, and test sets
#     test_split = 0.2
#     train_size = int((1 - validation_split - test_split) * len(full_dataset))
#     validation_size = int(validation_split * len(full_dataset))
#     test_size = len(full_dataset) - train_size - validation_size
#     train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

#     # DataLoader for training, validation, and test
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     return train_loader, val_loader, test_loader

def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')


    # iscal setting
    dailyActivity_merged_df = pd.read_csv("/home/ubuntu/isfolder/fl_agent_paper/fedops_iscal/iscal/data/dailyActivity_merged.csv")
    features = [
    'TotalSteps', 'TotalDistance', 'TrackerDistance', 'VeryActiveDistance',
    'ModeratelyActiveDistance', 'LightActiveDistance', 'VeryActiveMinutes',
    'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories'
    ]

    dataset_df = dailyActivity_merged_df[features]

    X_np = dataset_df.drop(columns=["Calories"], axis=1).to_numpy().astype(np.float32)
    y_np = dataset_df["Calories"].to_numpy().astype(np.float32)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=validation_split, random_state=seed)

    # 스케일링
    mean_train = X_train.mean(dim=0)
    std_train  = X_train.std(dim=0)
    std_train[std_train == 0] = 1.0  # 분산 0 방지
    def standardize(x, mean, std):
        return (x - mean) / std
    X_train = standardize(X_train, mean_train, std_train)
    X_val   = standardize(X_val,   mean_train, std_train)
    X_test  = standardize(X_test,  mean_train, std_train)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)
    


    return train_loader, val_loader, test_loader

# def gl_model_torch_validation(batch_size):
#     """
#     Setting up a dataset to evaluate a global model on the server
#     """
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
#     ])

#     # Load the test set of MNIST Dataset
#     val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)

#     # DataLoader for validation
#     gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

#     return gl_val_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    # iscal setting
    dailyActivity_merged_df = pd.read_csv("/app/code/dailyActivity_merged.csv")
    features = [
    'TotalSteps', 'TotalDistance', 'TrackerDistance', 'VeryActiveDistance',
    'ModeratelyActiveDistance', 'LightActiveDistance', 'VeryActiveMinutes',
    'FairlyActiveMinutes', 'LightlyActiveMinutes', 'SedentaryMinutes', 'Calories'
    ]

    dataset_df = dailyActivity_merged_df[features]

    X_np = dataset_df.drop(columns=["Calories"], axis=1).to_numpy().astype(np.float32)
    y_np = dataset_df["Calories"].to_numpy().astype(np.float32)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=seed)

    # 스케일링
    mean_train = X_train.mean(dim=0)
    std_train  = X_train.std(dim=0)
    std_train[std_train == 0] = 1.0  # 분산 0 방지
    def standardize(x, mean, std):
        return (x - mean) / std
    X_train = standardize(X_train, mean_train, std_train)
    X_val   = standardize(X_val,   mean_train, std_train)
    X_test  = standardize(X_test,  mean_train, std_train)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    test_dataset  = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, drop_last=False)

    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)


    return gl_val_loader