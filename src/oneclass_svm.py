import pandas as pd
import numpy as np

from sklearn.svm import OneClassSVM

from src.metrics import evaluate_predictions, save_metrics


def train_oneclass_svm() -> None:
    X_train = pd.read_csv("data/processed/X_train_normal.csv")
    X_val = pd.read_csv("data/processed/X_val.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")

    y_val = pd.read_csv("data/processed/y_val.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    model = OneClassSVM(
        kernel="rbf",
        nu=0.001,
        gamma=0.01
    )

    model.fit(X_train)

    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    best_threshold = -0.7536032653591321

    val_metrics = evaluate_predictions(y_val, val_scores, threshold=best_threshold)
    test_metrics = evaluate_predictions(y_test, test_scores, threshold=best_threshold)

    print("Validation metrics:")
    print(val_metrics)
    print("Test metrics:")
    print(test_metrics)

    save_metrics(test_metrics, model_name="oneclass_svm")
