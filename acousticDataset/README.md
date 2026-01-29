# Acoustic Anomaly Detection Module

This module implements an acoustic anomaly detection pipeline using the MIMII Fan dataset.  
The objective is to detect abnormal machine behavior using **only normal operating data**, following standard industrial condition-monitoring practice.

The development proceeded in **four clearly defined phases**, each addressing limitations observed in the previous stage.

---

## Why Autoencoders?

Autoencoders were chosen because:

- Abnormal data is **rare, incomplete, and non-exhaustive**
- The task is **anomaly detection**, not classification
- Autoencoders learn a **compact representation of normal behavior**
- Deviations from this learned manifold naturally indicate anomalies

Other approaches (classifiers, supervised CNNs) require labeled abnormal data and assume known failure modes, which is unrealistic in real industrial systems.

---

## Phase 1: Baseline Global Autoencoder

### Description
- Single autoencoder trained on **all normal samples**
- One log-mel spectrogram per audio file
- File-level reconstruction error used as anomaly score

### Results
- Mean ROC-AUC ≈ **~0.59**

### Limitations
- Strong overlap between normal and abnormal score distributions
- Anomalies are often **temporally localized**
- File-level averaging diluted short abnormal events
- Model learned **machine identity rather than health**

### Outcome
This phase established a working baseline but revealed the need for temporal modeling.

---

## Phase 2: Sliding-Window Global Model

### Description
- Audio files split into overlapping time windows
- Anomaly score computed per window
- File-level score aggregated from window scores

### Results
- ROC-AUC remained ≈ **~0.59**

### Limitations
- Training and inference distributions were misaligned
- Inter-machine acoustic variability dominated anomaly signal
- Sliding windows amplified machine-specific differences

### Outcome
This phase confirmed that **temporal resolution alone is insufficient** without accounting for asset variability.

---

## Phase 3: Sliding-Window Asset-Specific Models

### Description
- One autoencoder trained **per machine ID**
- Sliding-window training and inference
- Asset boundaries preserved

### Results
- Strong performance for some assets (AUC > 0.85)
- Weak performance for others (AUC ≈ 0.49–0.84)
- Mean ROC-AUC ≈ **0.62**

### Limitations
- Absolute spectral differences still dominated reconstruction loss
- Subtle anomalies remained hard to detect for certain assets

### Outcome
This phase demonstrated that **asset-aware modeling is necessary but not sufficient**.

---

## Phase 4: Normalized Sliding-Window Asset-Specific Models (Final)

### Description
- Per-asset Z-score normalization applied to log-mel features
- Normalization statistics computed from **normal training data only**
- Sliding-window inference with max aggregation

### Results
| Asset | ROC-AUC |
|------|--------|
| id_00 | ~0.62 |
| id_02 | ~0.95 |
| id_04 | ~0.78 |
| id_06 | ~0.96 |
| **Mean** | **~0.83** |

### Key Observations
- Strong improvement for acoustically informative assets
- One asset remained acoustically ambiguous
- Performance heterogeneity reflects **real-world fault observability**

### Outcome
This phase produced a **robust and realistic acoustic monitoring subsystem**, suitable for integration into a multimodal ensemble.

---

## Final Takeaway

- Acoustic anomaly detection is **highly effective for some assets**
- Performance varies due to intrinsic signal characteristics
- Asset-aware normalization is critical for practical deployment
- Acoustic monitoring alone is insufficient for complete coverage

This motivates the integration of **telemetry-based anomaly detection** and **ensemble fusion** in the full system.

---

## Status

✔ Acoustic subsystem finalized  
✔ No further tuning planned  
✔ Ready for multimodal fusion