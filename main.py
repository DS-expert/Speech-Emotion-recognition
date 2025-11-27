from src.preprocessing import preprocessing_pipeline, save_preprocessed_data
from config.config import PROCESSED_DATA

def main():

    # Load and Preprocessed the data

    X_train, X_test, y_train, y_test = preprocessing_pipeline()

    # Save the data of training and test dataset

    save_preprocessed_data(X_train, X_test, y_train, y_test, file_path=PROCESSED_DATA)

if __name__ == "__main__":
    main()