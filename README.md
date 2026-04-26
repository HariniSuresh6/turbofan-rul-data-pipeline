# turbofan-rul-data-pipeline
Scalable Data Pipeline for Turbofan Engine RUL Prediction with Activation-Based Drift Detection

## Overview
This project builds a scalable data pipeline to predict the Remaining Useful Life (RUL) of turbofan engines and monitor model reliability using activation-based drift detection.

## Features
- LSTM-based RUL prediction
- Activation-based drift detection (Jensen–Shannon divergence)
- Batch-wise streaming simulation
- Interactive dashboard using Streamlit

## Architecture
Data → Processing → LSTM Model → Drift Detection → Dashboard

## Tech Stack
Python, Pandas, NumPy, TensorFlow, PostgreSQL, SQLAlchemy, Streamlit

## Run
pip install -r requirements.txt  
python pipeline/lstm_drift.py  
python -m streamlit run dashboard/app.py  

## 📊 Output
- RUL predictions  
- Drift scores (Low / Medium / High)  
- Visual dashboard  

