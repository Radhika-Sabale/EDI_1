# EDI_1: Cloud-Resilient Illegal Mining Detection

Cloud-resilient illegal mining detection using multi-temporal satellite data and sequence modeling.

## 📌 Project Overview
Detecting unauthorized/illegal mining operations using single-frame satellite imagery is often confounded by seasonal agriculture, deforested land, and persistent cloud cover. This project overcomes these limitations through:
1. **Multi-Temporal Sequence Modeling:** Tracking locations over a 12-month period to distinguish persistent mining disturbance from seasonal agricultural cycles.
2. **Multi-Sensor Fusion (Sentinel-1 & Sentinel-2):** Combining optical Sentinel-2 (NDVI for vegetation degradation) with synthetic aperture radar (SAR) Sentinel-1 (VH backscatter to penetrate cloud cover).
3. **Sequence Classification Pipeline:** Preparing normalized multi-sensor time-series feature sequences ready for attention-based temporal modeling and comparative baseline models.

---

## 📂 Project Structure

```text
Illegal Mining Detection/
├── Data/
│   ├── 01_get_mining_sites.py              # Fetches and filters ground-truth mining locations
│   ├── 02_get_negative_sites.py            # Generates balanced non-mining / negative background sites
│   ├── 03_combine_locations.py             # Merges and indexes input locations with labels
│   ├── 04_gee_extract_timeseries.py        # Extracts 12-month Sentinel-1/2 time series via Google Earth Engine
│   ├── 05_clean_and_reshape.py             # Quality checks, reshapes time series into array sequences
│   ├── 06_replace_zero_coverage_sites.py   # Diagnoses and replaces cloudy/zero-coverage coordinates
│   ├── 07_compute_derived_features.py      # Computes delta features, rolling stats, and feature engineering
│   ├── 08_train_test_split.py              # Splits dataset into stratified train/val/test splits
│   ├── 09_metrics_pipeline.py              # Evaluates sequence metrics and model performance
│   ├── 10_baseline_classical.py            # Classical baselines: Logistic Regression & Random Forest
│   ├── 11_baseline_lstm.py                 # Plain sequence baseline: LSTM model without attention
│   ├── connect.py                          # Earth Engine connectivity verification script
│   ├── Dedupe timeseries.py                # Time-series deduplication utility
│   ├── diagnose_vh_gap.py                  # SAR Sentinel-1 VH gap diagnostic script
│   ├── purge_incomplete_rows.py            # Cleans incomplete time series sequences
│   ├── info                                # Project reference and registry documentation notes
│   ├── input_locations.csv                 # Consolidated target coordinates (mining + negative)
│   ├── mining_sites.csv                    # Filtered mining coordinates
│   ├── negative_sites.csv                  # Filtered negative coordinates
│   ├── replacement_sites.csv               # Supplementary replacement coordinates
│   ├── timeseries_data.csv                 # Raw monthly multi-sensor time series
│   ├── cleaned_sequences.npz               # Cleaned time-series array sequences
│   ├── features_sequences.npz              # Extracted feature tensors
│   ├── scaler_params.npz                   # Fitted feature scaling parameters
│   └── train_test_split.npz                # Stratified train/val/test dataset split
├── Model/
│   ├── 01_attention_fusion.py              # Main LSTM + Temporal Attention model & attention weight extractor
│   ├── 02_multiseed_evaluation.py          # 10-repeat stratified multi-seed benchmark across all 4 initial models
│   ├── 03_handcrafted_features_baselines.py# 22 hand-crafted summary features evaluated on LR & RF
│   ├── 04_tune_attention_model.py          # 5-fold CV tuning of LSTM & Transformer architectures on training data
│   ├── 05_seed_ensemble.py                 # 5-model seed ensemble for variance reduction and probability smoothing
│   ├── plot_attention.py                   # Generates temporal attention distribution plots
│   ├── attention_plot.png                  # Visualized attention curves (mining vs non-mining)
│   ├── attention_weights_test.npz          # Extracted per-month attention weights for test locations
│   ├── multiseed_results.csv               # Detailed 10-seed metrics for initial models
│   ├── multiseed_summary.csv               # Aggregated mean +/- std benchmark summary
│   ├── handcrafted_results.csv             # Detailed results for handcrafted feature baselines
│   ├── handcrafted_summary.csv             # Summary table for handcrafted feature baselines
│   ├── tuning_cv_results.csv               # 5-fold CV hyperparameter & architecture search results
│   ├── tuned_model_evaluation.csv          # Single test set evaluation of the winning tuned model
│   ├── tuned_multiseed_summary.csv         # 10-seed repeat summary for the winning tuned model
│   ├── ensemble_results.csv                # Detailed 10-seed metrics comparing single vs ensemble
│   └── ensemble_summary.csv                # Win/loss/tie and mean +/- std summary for the seed ensemble
├── .gitignore
└── README.md
```

---

## 🚀 Pipeline Workflow

1. **Location Sampling (`01_get_mining_sites.py` & `02_get_negative_sites.py`)**  
   Collects verified global mining coordinates and generates geographically matched non-mining negative sites.

2. **Dataset Indexing (`03_combine_locations.py`)**  
   Consolidates target points into `input_locations.csv` with unique location identifiers and class labels.

3. **Earth Engine Time-Series Extraction (`04_gee_extract_timeseries.py`)**  
   Batch-extracts 12-month monthly Sentinel-2 NDVI and Sentinel-1 SAR VH backscatter composite measurements.

4. **Cleaning & Reshaping (`05_clean_and_reshape.py` & `06_replace_zero_coverage_sites.py`)**  
   Handles missing values, removes incomplete series, and formats data into standard 3D sequence tensors `(samples, timesteps, features)`.

5. **Feature Engineering & Splits (`07_compute_derived_features.py` & `08_train_test_split.py`)**  
   Computes temporal differences ($\Delta\text{NDVI}$, $\Delta\text{VH}$), normalizes features, and exports stratified train/test arrays.

6. **Model Training & Evaluation (`09_metrics_pipeline.py`)**  
   Evaluates sequence classification performance across temporal attention models and baseline architectures.

---

## 🛠️ Prerequisites & Setup

1. **Python Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install earthengine-api pandas numpy scikit-learn
   ```

2. **Google Earth Engine Authentication:**
   ```bash
   earthengine authenticate
   ```
   Ensure your GEE project ID is properly configured (e.g., in `connect.py` or script configurations).
