import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def evaluate_predictions(y_true: np.ndarray, scores: np.ndarray, threshold: float = None) -> dict:

    if threshold is None:
        threshold = np.percentile(scores, 99)

    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }

    return metrics


def save_metrics(metrics: dict, model_name: str, path: str = "results/metrics.csv") -> None:
    df = pd.DataFrame([metrics])
    df["model"] = model_name

    try:
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(path, index=False)