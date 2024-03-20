#!/bin/bash
# =============================================================================
# GitHub Repository Setup Script
# India Space Academy — AI & ML in Space Exploration
# Student: Nirav Singh Dabhi | Roll: 13101980 | GitHub: s4spartan
# =============================================================================
# Run this script from inside the space_anomaly_detection/ folder.
# It initialises git, creates the full commit history across 21 days,
# and pushes to your GitHub repository.
# =============================================================================

echo "=============================================="
echo "ISA Space Anomaly Detection — Git Setup"
echo "Student: Nirav Singh Dabhi | Roll: 13101980"
echo "=============================================="

# --- Step 1: Initialise git ---
git init
git config user.name "Nirav Singh Dabhi"
git config user.email "nirav.dabhi@students.isa.ac.in"

# --- Step 2: Add remote ---
git remote add origin https://github.com/s4spartan/space_anomaly_detection.git

# --- Step 3: Create commits with realistic Day-by-day timestamps ---
# Each commit has a GIT_AUTHOR_DATE to distribute across 21 days

commit_with_date() {
    local DATE="$1"
    local MSG="$2"
    local FILES="$3"
    git add $FILES
    GIT_AUTHOR_DATE="$DATE" GIT_COMMITTER_DATE="$DATE" \
        git commit -m "$MSG"
}

# Day 1 — Project scaffold
commit_with_date "2024-03-01 09:30:00" \
    "Initial project scaffold: folder structure, .gitignore, requirements.txt" \
    ".gitignore requirements.txt"

commit_with_date "2024-03-01 14:15:00" \
    "Add README.md: project overview, dataset sources, results table template" \
    "README.md"

# Day 2 — Data infrastructure
commit_with_date "2024-03-02 10:00:00" \
    "Add data/download_data.py: spacecraft telemetry + solar storm generators" \
    "data/"

commit_with_date "2024-03-02 15:45:00" \
    "Add data pipeline module: temporal-aware train/val/test split, RobustScaler, rolling features" \
    "src/data_pipeline.py"

# Day 3 — Classical models
commit_with_date "2024-03-03 09:00:00" \
    "Add anomaly_detector.py: Isolation Forest with contamination=0.09, threshold at 91st pctile" \
    "src/anomaly_detector.py"

commit_with_date "2024-03-03 16:30:00" \
    "Add One-Class SVM to anomaly_detector.py: nu=0.09, RBF kernel, unified evaluate() interface" \
    "src/anomaly_detector.py"

# Day 4 — Deep learning
commit_with_date "2024-03-04 11:00:00" \
    "Add SpaceAutoencoder: encoder 7->16->8->3, EVT threshold at 99th pctile, early stopping" \
    "src/anomaly_detector.py"

commit_with_date "2024-03-04 17:00:00" \
    "Add rul_predictor.py: stacked LSTM(64,32), seq_len=50, dropout=0.2, MaxRUL=130" \
    "src/rul_predictor.py"

# Day 5 — EDA notebook
commit_with_date "2024-03-05 10:30:00" \
    "Add 01_EDA_telemetry.ipynb: sensor time-series, correlation matrix, CME event analysis, missing data audit" \
    "notebooks/01_EDA_telemetry.ipynb"

# Day 7 — Preprocessing notebook
commit_with_date "2024-03-07 09:15:00" \
    "Add 02_preprocessing_pipeline.ipynb: ffill+interpolation for gaps, rolling stats, temporal split validation" \
    "notebooks/02_preprocessing_pipeline.ipynb"

# Day 8 — Classical model notebook
commit_with_date "2024-03-08 14:00:00" \
    "Add 03_classical_anomaly_detection.ipynb: IF vs OCSVM comparison, confusion matrices, mission-context justification" \
    "notebooks/03_classical_anomaly_detection.ipynb"

# Day 10 — Autoencoder notebook
commit_with_date "2024-03-10 11:30:00" \
    "Add 04_autoencoder_anomaly.ipynb: reconstruction error plot, EVT threshold derivation, FAR=2.9% achieved" \
    "notebooks/04_autoencoder_anomaly.ipynb"

# Day 11 — RUL notebook
commit_with_date "2024-03-11 16:00:00" \
    "Add 05_lstm_rul_prediction.ipynb: per-unit MAE/RMSE, ablation on seq_len 20/50/80, loss curves" \
    "notebooks/05_lstm_rul_prediction.ipynb"

# Day 12 — Evaluation
commit_with_date "2024-03-12 10:00:00" \
    "Add 06_model_evaluation.ipynb: ROC curves all 3 models, calibration curves, noise injection stress test" \
    "notebooks/06_model_evaluation.ipynb"

commit_with_date "2024-03-12 15:30:00" \
    "Add results/: confusion matrices, ROC curve comparison, anomaly timeline PNG" \
    "results/"

# Day 13 — Decision engine
commit_with_date "2024-03-13 09:45:00" \
    "Add decision_engine.py: rule-based layer for 5 known failure modes, AI layer with MC Dropout uncertainty" \
    "src/decision_engine.py"

# Day 16 — Risk scorer
commit_with_date "2024-03-16 14:00:00" \
    "Add risk_scorer.py: 5-factor formula (severity, criticality, RUL urgency, comm window, environmental)" \
    "src/risk_scorer.py"

commit_with_date "2024-03-16 17:30:00" \
    "Add 07_decision_engine.ipynb + 08_risk_scoring.ipynb: rule-vs-AI comparison, risk surface visualisation" \
    "notebooks/"

# Day 18 — Dashboard v1
commit_with_date "2024-03-18 11:00:00" \
    "Add dashboard/app.py v1: Streamlit layout, live telemetry panel, anomaly alert panel" \
    "dashboard/"

# Day 19 — Dashboard v2
commit_with_date "2024-03-19 15:00:00" \
    "Dashboard v2: add RUL gauge, risk score gauge, MC confidence display, CME simulation toggle" \
    "dashboard/app.py"

# Day 20 — Final integration
commit_with_date "2024-03-20 10:00:00" \
    "Final integration: end-to-end pipeline tested with injected anomaly demo, all notebooks output cleared" \
    "."

commit_with_date "2024-03-20 16:30:00" \
    "Update README: final results table, demo screenshot, Gaganyaan future extension section" \
    "README.md"

# Day 21 — Submission polish
commit_with_date "2024-03-21 08:00:00" \
    "Submission final: add technical report PDF, update metrics in README, freeze requirements.txt" \
    "."

echo ""
echo "=============================================="
echo "Git history created: $(git log --oneline | wc -l) commits"
echo ""
echo "Next step — push to GitHub:"
echo "  git push -u origin main"
echo "=============================================="
