import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

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

def XGB_model(X_train, y_train, X_test):
    """
    XGBoost Model training on training features and labels
    
    :params -> 
        X_train: training features
        y_train: training labels
    :return:
        model: trained XGBoost model
    """

    xgb = XGBClassifier()

    xgb.fit(X_train, y_train)

    # Predict
    y_pred = xgb.predict(X_test)

    return xgb, y_pred

def evalution_loop(model, train_dataloader, test_dataloader, loss_func, optimizer, epochs, random_state=42, device='cpu'):
    """
    Evalution loop for CNN Models
    :params
        model: CNN model
        train_dataloader: DataLoader for traning data
        test_dataloader: DataLoader for testing data
        loss_func: loss function used for CNN model
        optimimzer: Optimizer usef for CNN Model
        random_state: random state for reproducibility
        device: device to run the model on (cpu or cuda)
        epochs: number of epochs to train the model
    :returns
        return train_losses, test_losses, train_accuracy_score, test_accuracy_score
    """

    torch.manual_seed(random_state)

    model.to(device)

    train_losses = []
    test_losses = []
    train_accuracy_score = []
    test_accuracy_score = []

    for epoch in range(epochs):
        # Training of the model
        model.train()

        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        # Forward pass

        for X_batch, y_batch in train_dataloader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_logits = model(X_batch)
            y_pred = torch.argmax(y_logits, dim=1)

            loss = loss_func(y_logits, y_batch)
            train_acc = accuracy_score(y_batch, y_pred)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            # Accumalate loss
            total_train_loss += loss.item() * X_batch.size(0)
            total_train_correct += (y_pred == y_batch).sum().item()
            total_train_samples += X_batch.size(0)
        
        # Evaluation of Model
        model.eval()

        total_test_loss = 0
        total_test_correct = 0
        total_test_samples = 0

        with torch.inference_mode():
            for X_batch, y_batch in test_dataloader:

                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                test_logits = model(X_batch)
                test_preds = torch.argmax(test_logits, dim=1)
                
                test_loss = loss_func(test_logits, y_batch)
                test_acc = accuracy_score(y_batch, test_preds)

                total_test_loss += test_loss.item() * X_batch.size(0)
                total_test_correct += (test_preds == y_batch).sum().item()
                total_test_samples += X_batch.size(0)

        # Calculate average losses and accuracies
        avg_train_loss = total_train_loss / total_train_samples
        avg_test_loss = total_train_loss / total_test_samples

        avg_train_acc = total_train_correct / total_train_samples * 100
        avg_test_acc = total_test_correct / total_test_samples * 100
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracy_score.append(avg_train_acc)
        test_accuracy_score.append(avg_test_acc)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}"
                  f"Train Loss: {avg_train_loss:.4f} | Train Accuracy : {avg_train_acc:.2f}% ||"
                  f"Test Loss: {avg_test_loss:.4f} | Test Accuracy : {avg_test_acc:.2f}%")

    return train_losses, test_losses, train_accuracy_score, test_accuracy_score
