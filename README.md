# Capstone Project: Multimodal Predictive Maintenance

This repository contains a capstone implementation of a multimodal health monitoring system that combines:

- Telemetry-based Remaining Useful Life (RUL) estimation on NASA CMAPSS engine data
- Acoustic anomaly detection on MIMII fan recordings
- A fusion model that combines both signals into a final decision layer

The project is organized as notebook-first research code with saved model artifacts and intermediate arrays.

## Repository Modules

- `acousticDataset/`
  - Acoustic anomaly detection experiments and final subsystem using autoencoders on MIMII.
  - Includes an existing detailed README at `acousticDataset/README.md`.
- `telemetryDataset/` (your "TelementryData" module)
  - Telemetry preprocessing and model training for FD001-FD004 subsets.
  - Includes both global and fault-specific (router + specialists) pipelines.
- `Multimodal_Fusion/` (fusion model module)
  - Meta-learner that combines telemetry prediction with acoustic anomaly scores.
  - Includes SHAP-based explainability notebook.
- `FastF1_demo/`
  - Separate demonstration notebook using FastF1 telemetry data.

## Datasets

- Shared dataset bundle used in this project:
  - [Google Drive folder](https://drive.google.com/drive/folders/1FcnvRi5x1qt9Hbm1psCnYl5lTygc8iF0?usp=share_link)
- MIMII acoustic dataset:
  - [Zenodo: MIMII Dataset](https://zenodo.org/records/3384388)
- Telemetry notebooks are built around NASA CMAPSS FD subsets (`FD001` to `FD004`).

## End-to-End Workflow

1. Train/evaluate acoustic subsystem in `acousticDataset/`.
2. Train/evaluate telemetry models in `telemetryDataset/`.
3. Generate/load acoustic scores (`mimii_normal_scores.npy`, `mimii_abnormal_scores.npy`).
4. Run fusion notebook in `Multimodal_Fusion/` to train final multimodal model.
5. Use SHAP notebook for interpretability analysis.

## Documented Outputs Snapshot

The following results are consolidated from `Documentation of PJT2 NASA.docx`.

| Stage | Key Result |
|---|---|
| FD001 Global Attention-LSTM | RMSE: **14.6955**, NASA Score: **307.58**, Accuracy (±15 cycles): **66.00%** |
| FD002 Global Attention-LSTM | Test RMSE: **27.0426** |
| FD003 Global Baseline | Test RMSE: **21.04** |
| FD003 MoE (Router + Specialists) | Final RMSE: **20.24** (improved vs global baseline) |
| FD004 Global Baseline | Test RMSE: **40.56** |
| FD004 Asymmetric MoE | Final RMSE: **41.51** (diagnostic interpretability trade-off) |
| Multimodal Fusion (Linear Meta-Learner) | RMSE: **39.13**, MAE: **28.45** |
| Telemetry-Only FD004 MoE (for fusion comparison) | RMSE: **41.51**, MAE: **32.12** |

## Graph Highlights

### Acoustic Score Distribution (MIMII)

![Acoustic anomaly score distribution](docs/images/notebooks/acousticdataset_global_model_acoustic_anomaly_detection_autoencoder_model_ipynb_img01.png)

### FD003 Fault Clustering (Unsupervised)

![FD003 fault clustering](docs/images/notebooks/telemetrydataset_models_fault_specific_fd003_31_fault_clustering_pre_ipynb_img01.png)

### SHAP Explainability for Fusion

![Fusion SHAP summary](docs/images/notebooks/multimodal_fusion_explainable_ai_shap_ipynb_img01.png)

## Environment Notes

The notebooks import the following core libraries:

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `joblib`
- `tensorflow` / `keras`
- `shap` (fusion explainability)
- `librosa` (acoustic preprocessing)
- `fastf1` (demo only)

Many notebooks currently contain local Windows-style paths such as `D:\\PJT2\\...`.
If you run them on another machine, update those path constants before execution.

## Key Outputs Already Present

- Telemetry:
  - Saved checkpoints under `telemetryDataset/models/**/model_checkpoints/`
  - Processed arrays and scalers under `telemetryDataset/processed_data/`
- Fusion:
  - `Multimodal_Fusion/linear_multimodal_fusion.pkl`
  - `Multimodal_Fusion/X_train_fusion.npy`
  - `Multimodal_Fusion/X_test_fusion.npy`
- Acoustic:
  - `acousticDataset/asset-specific/mimii_normal_scores.npy`
  - `acousticDataset/asset-specific/mimii_abnormal_scores.npy`

## Module READMEs

- Telemetry module README: `telemetryDataset/README.md`
- Fusion module README: `Multimodal_Fusion/README.md`
- Acoustic module README: `acousticDataset/README.md`
