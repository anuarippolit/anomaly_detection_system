import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_data(output_dir="data/processed", random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    import kagglehub

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    df = pd.read_csv(path + "/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split train/val/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=random_state
    )

    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Log transform Amount
    for dataset in [X_train, X_val, X_test]:
        dataset["Amount"] = np.log1p(dataset["Amount"])


    X_train_normal = X_train[y_train == 0]

    scaler = StandardScaler()
    scaler.fit(X_train_normal)

    X_train_normal_scaled = scaler.transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)


    feature_names = X.columns

    X_train_normal_scaled = pd.DataFrame(X_train_normal_scaled, columns=feature_names)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)

    X_train_normal_scaled.to_csv(f"{output_dir}/X_train_normal.csv", index=False)
    X_val_scaled.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test_scaled.to_csv(f"{output_dir}/X_test.csv", index=False)

    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    joblib.dump(scaler, f"{output_dir}/scaler.joblib")

    print("Preprocessing completed and saved to data/processed/")

if __name__ == "__main__":
    preprocess_data()