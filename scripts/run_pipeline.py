from src.preprocessing import preprocess_data
from src.isolation_forest import IsolationForest
# from src.oneclass_svm import train_oneclass_svm
# from src.autoencoder import train_autoencoder
from src.metrics import evaluate_predictions, save_metrics
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np

def best_threshold_by_f1(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    return thresholds[np.argmax(f1_scores[:-1])]

def run_isolation_forest(X_train, X_val, X_test, y_val, y_test):
    print("Training Isolation Forest...")
    clf = IsolationForest(
        n_estimators=100,
        max_samples=1024,
        contamination="auto",
        max_features=1.0,
        random_state=42
    )
    clf.fit(X_train)

    # high scores = anomalous unlike from sklearn
    val_scores = clf.score_samples(X_val)
    threshold = best_threshold_by_f1(y_val, val_scores)

    test_scores = clf.score_samples(X_test)
    metrics = evaluate_predictions(y_test, test_scores, threshold)

    print(f"Isolation Forest: {metrics}")
    save_metrics(metrics, model_name="IsolationForest")

def main() -> None:

    print("Preprocessing...")
    preprocess_data()

    print("Loading data...")
    X_train = pd.read_csv('data/processed/X_train_normal.csv').values
    X_val   = pd.read_csv('data/processed/X_val.csv').values
    X_test  = pd.read_csv('data/processed/X_test.csv').values
    y_val   = pd.read_csv('data/processed/y_val.csv').values.flatten()
    y_test  = pd.read_csv('data/processed/y_test.csv').values.flatten()

    run_isolation_forest(X_train, X_val, X_test, y_val, y_test)

    # run_oneclass_svm(X_train, X_val, X_test, y_val, y_test)
    # run_autoencoder(X_train, X_val, X_test, y_val, y_test)

    print("Pipeline finished. Results saved to results/metrics.csv")

if __name__ == "__main__":
    main()