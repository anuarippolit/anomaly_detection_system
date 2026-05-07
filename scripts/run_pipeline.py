from src.preprocessing import preprocess_data
# from src.isolation_forest import train_isolation_forest
# from src.oneclass_svm import train_oneclass_svm
from src.autoencoder import train_autoencoder


def main() -> None:

    print("Preprocessing...")
    preprocess_data()

    #print("Training Isolation Forest")
    #train_isolation_forest()

    # print("Training One-Class SVM")
    # train_oneclass_svm()

    print("Training Autoencoder")
    train_autoencoder()

    print("Pipeline finished.")


if __name__ == "__main__":
    main()