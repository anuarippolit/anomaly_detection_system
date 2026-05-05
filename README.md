# Anomaly Detection System

Project goal: compare three anomaly detection models on the credit card fraud dataset:

- Isolation Forest  
- One-Class SVM  
- Autoencoder  

The dataset is downloaded automatically using `kagglehub`.

## Setup

git clone https://github.com/anuarippolit/anomaly_detection_system  
cd anomaly_detection_system  
pip install -r requirements.txt or make setup  

## Run

python scripts/run_pipeline.py or make run  

Pipeline does:
1. Downloads dataset from Kaggle  
2. Runs preprocessing  
3. Saves processed data to data/processed/  
4. Runs models that are NOT commented out (if you want to try some mode, remove the comment) 

## Project structure

configs/        optional config files  
data/           raw/processed data (not committed, run preprocessing as mentioned in Run step)  
notebooks/      EDA and preprocessing exploration  
scripts/        main pipeline 
src/            preprocessing, metrics, models  
tests/          simple tests  
results/        metrics, plots, saved models  

## Processed data

data/processed/X_train_normal.csv  
data/processed/X_val.csv  
data/processed/X_test.csv  
data/processed/y_val.csv  
data/processed/y_test.csv  
data/processed/scaler.joblib  

Load data and use it as:

import pandas as pd  

X_train = pd.read_csv("data/processed/X_train_normal.csv")  
X_val = pd.read_csv("data/processed/X_val.csv")  
X_test = pd.read_csv("data/processed/X_test.csv")  

y_val = pd.read_csv("data/processed/y_val.csv").squeeze()  
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()  

Important:  
- Train ONLY on X_train_normal  
- Validate/Test on X_val / X_test  

## How to implement your model

Each person works in ONE file:

src/isolation_forest.py  
src/oneclass_svm.py  
src/autoencoder.py  

Create function:

def train_your_model() -> None:  
    pass  

Inside:
1. Load processed data  
2. Train on X_train_normal  
3. Generate anomaly scores on test/val   
4. Use metrics  
5. Save results  

## Metrics usage

from src.metrics import evaluate_predictions, save_metrics  

metrics = evaluate_predictions(  
    y_true=y_test,  
    scores=test_scores  
)  

save_metrics(metrics, model_name="your_model_name")  

IMPORTANT:  
scores must be: higher = more anomalous  

If opposite:  
scores = -scores  

## TODO

1. Please someone check metrics.py
2. Implement model, and try 
3. Make a seperate branch for your commit please, do not push into main branch plssssss