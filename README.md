# AI-Based Real-Time Anomaly Detection & Autonomous Decision Support System for Deep Space Missions

**India Space Academy — AI & ML in Space Exploration**
**Student:** Nirav Singh Dabhi | **Roll No:** 13101980 | **Level:** PG
**GitHub:** [@s4spartan](https://github.com/s4spartan)

---

## Project Overview

Deep space missions such as Chandrayaan-3, Aditya-L1, and Mars Perseverance face communication delays of 4–24 minutes (one-way), making real-time ground intervention impossible during anomaly windows. This project builds an end-to-end AI system that:

- Detects spacecraft subsystem anomalies from simulated telemetry and solar storm data
- Predicts Remaining Useful Life (RUL) of critical subsystems
- Makes autonomous corrective action recommendations with confidence scoring
- Visualises mission health in a real-time dashboard

---

## Results Summary

| Model | Dataset | F1-Score | ROC-AUC | False Alarm Rate | RUL MAE |
|-------|---------|----------|---------|-----------------|---------|
| Isolation Forest | Spacecraft Telemetry | 0.84 | 0.89 | 6.2% | — |
| One-Class SVM | Spacecraft Telemetry | 0.81 | 0.86 | 8.1% | — |
| Autoencoder (Deep Learning) | Spacecraft Telemetry | 0.91 | 0.94 | 3.8% | — |
| LSTM (RUL Prediction) | Simulated Degradation | — | — | — | 12.4 cycles |
| Ensemble (Final) | Combined | **0.93** | **0.96** | **2.9%** | **11.1 cycles** |

> Best model: Deep Learning Autoencoder + LSTM Ensemble. Selected on the basis of lowest false alarm rate — critical for autonomous space operations where false positives trigger irreversible corrective actions.

---

## Repository Structure

```
space_anomaly_detection/
├── data/
│   ├── download_data.py          # Scripts to fetch NASA SMAP/MSL and simulated telemetry
│   └── README_data.md            # Dataset descriptions and sources
├── notebooks/
│   ├── 01_EDA_telemetry.ipynb             # Phase 1: Exploratory data analysis
│   ├── 02_preprocessing_pipeline.ipynb    # Phase 2: Feature engineering & cleaning
│   ├── 03_classical_anomaly_detection.ipynb  # Phase 2: Isolation Forest, One-Class SVM
│   ├── 04_autoencoder_anomaly.ipynb       # Phase 2: Deep Learning Autoencoder
│   ├── 05_lstm_rul_prediction.ipynb       # Phase 2: LSTM for RUL prediction
│   ├── 06_model_evaluation.ipynb          # Phase 2: ROC-AUC, confusion matrix, FAR
│   ├── 07_decision_engine.ipynb           # Phase 3: Autonomous decision layer
│   └── 08_risk_scoring.ipynb              # Phase 3: Multi-factor risk model
├── src/
│   ├── anomaly_detector.py       # Core anomaly detection module
│   ├── decision_engine.py        # Autonomous decision & corrective action module
│   ├── risk_scorer.py            # Multi-factor risk scoring system
│   ├── rul_predictor.py          # RUL prediction using LSTM
│   ├── data_pipeline.py          # End-to-end preprocessing pipeline
│   └── utils.py                  # Shared utilities
├── models/
│   ├── autoencoder_v1_auc0.94.h5
│   ├── lstm_rul_v1_mae12.4.pkl
│   └── isolation_forest_v1.pkl
├── results/
│   ├── confusion_matrix_autoencoder.png
│   ├── confusion_matrix_isolation_forest.png
│   ├── roc_curve_comparison.png
│   ├── rul_prediction_plot.png
│   ├── anomaly_timeline.png
│   └── metrics_summary.csv
├── dashboard/
│   └── app.py                    # Streamlit mission control dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/s4spartan/space_anomaly_detection.git
cd space_anomaly_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python data/download_data.py

# 4. Run the dashboard
streamlit run dashboard/app.py
```

---

## Dataset Sources

| Dataset | Source | Usage |
|---------|--------|-------|
| Simulated Spacecraft Subsystem Telemetry | Generated via OpenMCT simulation framework | Anomaly detection training |
| Solar Storm / Space Weather Data (OMNI) | NASA CDAWeb public archive | Aditya-L1 context, environmental classification |
| SMAP Anomaly Dataset | NASA JPL (public) | Labelled anomaly validation |
| MSL (Mars Science Laboratory) Anomaly Dataset | NASA JPL (public) | Cross-mission generalisation test |

---

## Mission Context

This system is designed with the following real mission constraints:

- **Chandrayaan-3 / Pragyan**: 14-day lunar day cycles, intermittent communication, thermal cycling anomalies
- **Aditya-L1**: Solar wind and UV flux monitoring; sensor saturation during CME events classified as environment (not fault)
- **Mars missions**: 4–24 min one-way communication delay; all corrective actions must be autonomous

---

## Future Mission Extension

This system is designed for direct extension to **ISRO's Gaganyaan human spaceflight mission**. Life support telemetry (cabin pressure, O₂ concentration, CO₂ scrubber performance) follows the same multivariate time-series structure as spacecraft propulsion telemetry. The autonomous decision layer can be retrained on Gaganyaan subsystem criticality weights where a false negative (missed anomaly) is unacceptable. See Section 8 of the technical report for full extension specification.

---

## Technical Report

[`report/ISA_Technical_Report_NiravSinghDabhi_13101980.pdf`](report/)

---

*Vikasit Bharat @2047 — Atmanirbhar, Sashakt aur Unnata Bharat*
*India Space Academy | www.isa.ac.in*
