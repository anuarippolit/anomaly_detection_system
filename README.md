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
4. Runs models

The flow: 
1. make setup
2. make run

For test: make test

## Project structure

configs/        optional config files  
data/           raw/processed data (not committed, run preprocessing as mentioned in Run step)  
notebooks/      EDA and preprocessing exploration  
scripts/        main pipeline 
src/            preprocessing, metrics, models  
tests/          simple tests  
results/        metrics, plots, saved models  


## Notes

- Processed datasets are not committed to Git.
- Run preprocessing before training models.
- Results are stored inside the `results/` directory.
