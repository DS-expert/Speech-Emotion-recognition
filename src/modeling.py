import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

## CNN

def dataset_to_tensor(X_train, y_train, X_test, y_test):

    """
    Docstring for dataset_to_tensor
    
    :param X_train: training features 
    :param y_train: trainin labels
    :param X_test: testing features
    :param y_test: testing labels

    return: Convert dataset into tensors
    """

    X_train_tensor = torch.from_numpy(X_train)
    X_test_tensor = torch.from_numpy(X_test)
    y_train_tensor = torch.from_numpy(y_train)
    y_train_tensor = y_train_tensor.long()
    y_test_tensor = torch.from_numpy(y_test).long()

    # Unsqueeze the X_train and X_test 
    X_train_tesnor = X_train_tensor.unsqueeze(1)
    X_test_tensor = X_test_tensor.unsqueeze(1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor