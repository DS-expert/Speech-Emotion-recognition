import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


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

    for epoch in tqdm(range(epochs), desc='Training Epochs'):
        # Training of the model
        model.train()

        total_train_loss = 0
        total_train_correct = 0
        total_train_samples = 0

        # Forward pass

        for X_batch, y_batch in tqdm(train_dataloader, desc="Training Batch Data"):

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
            for X_batch, y_batch in tqdm(test_dataloader, desc="Testing Batch Test data"):

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
        avg_test_loss = total_test_loss / total_test_samples

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the given model on the test dataset and return accuracy, ROC AUC Score and classification report.

    Parameters:
    model: Trained machine learning model
    X_test: Feature of the test dataset
    y_test: True labels of the test dataset

    *returns: accuracy_score, roc_auc_score, classification_report
    """
    y_pred = model.predict(X_test)

    acc_score = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    return {
        "Accuracy Score": acc_score,
        "ROC AUC Score ": roc,
        "Classification Report" : cr
    }