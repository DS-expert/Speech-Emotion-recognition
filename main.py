from src.preprocessing import preprocessing_pipeline, save_preprocessed_data
from config.config import PROCESSED_DATA
from src.modeling import (
    dataset_to_tensor, create_dataloaders, CNNModel, save_model
)
from src.evaluation import evalution_loop
import torch.nn as nn
import torch.optim as optim
from config.config import RESULT_DIR
from config.config import X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
from src.utils.logger import get_logger
import joblib

logger = get_logger(__name__)

def main():

    # Load and Preprocessed the data
    if not X_TRAIN.exists() and not X_TEST.exists() and not Y_TRAIN.exists() and not Y_TEST.exists():
        logger.info("Preprocessing the data...")  
        X_train, X_test, y_train, y_test = preprocessing_pipeline()

        # Save the data of training and test dataset

        save_preprocessed_data(X_train, X_test, y_train, y_test, file_path=PROCESSED_DATA)
    
    else:
        logger.info("Loading preprocessed data...")

        X_train = joblib.load(X_TRAIN)
        X_test = joblib.load(X_TEST)
        y_train = joblib.load(Y_TRAIN)
        y_test = joblib.load(Y_TEST)

    # 2. Prepare for CNN
    X_tr_t, y_tr_t, X_te_t, y_te_t = dataset_to_tensor(X_train, y_train, X_test, y_test)
    train_loader, test_loader = create_dataloaders(X_tr_t, y_tr_t, X_te_t, y_te_t)

    # 3. Initialize Model
    model = CNNModel(num_classes=8)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train
    logger.info("Starting training...")
    evalution_loop(model, train_loader, test_loader, loss_fn, optimizer, epochs=30)

    # 5. SAVE THE MODEL (Crucial for Streamlit)
    save_model(model, path=RESULT_DIR, model_name="emotion_cnn", deep_learning_model=True)
    logger.info("Model saved to models/emotion_cnn.h5")

if __name__ == "__main__":
    main()