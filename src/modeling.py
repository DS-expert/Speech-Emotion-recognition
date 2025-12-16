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


def create_dataloaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=32, shuffle=False):
    """
    create dataloaders for training and testing

    **params:
        X_train_tesnor: training features tensor
        X_test_tensor: testing_features_tensor
        y_train_tensor: training labels tensor
        y_test_tensor: testing labels tensor
        batch_size: batch size for dataloader
        shuffle: Whether to shuffle the data or not
    
    return: 
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def flat_data(X_train, X_test):
    """
    Flatten the data for classic model

    **params:
        X_train: training features
        X_test: testing features
    return:
        X_train_flat: flattened training features
        X_test_flat: flattened testing features
    """

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    return X_train_flat, X_test_flat

class CNNModel(nn.Module):

    def __init__(self, num_classes=8, kernel_size=3, padding=1, dropout_rate=0.3):

        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 4 * 37, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, X):
        X = self.conv_layers(X)
        X = torch.flatten(X, 1)
        X = self.fc_layer(X)

        return X    