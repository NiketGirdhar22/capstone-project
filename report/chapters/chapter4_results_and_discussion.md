# Chapter 4: Results and Discussion

## 4.1 Evaluation Strategy and Experimental Protocol
This chapter reports stage-wise outcomes aligned with the methodology in Chapter 3. Results are organized by subsystem first (telemetry, acoustic), then integrated fusion analysis.

### 4.1.1 Why Stage-Wise Evaluation Was Necessary
A single end metric cannot reveal where improvements originate. Stage-wise evaluation was used to answer four concrete questions:
1. How strong are global telemetry baselines per CMAPSS subset?
2. Do specialist/routed telemetry models provide measurable gains?
3. Which acoustic modeling intervention contributes the largest AUC gain?
4. Does multimodal fusion improve over telemetry-only baseline with interpretable behavior?

### 4.1.2 Metric Set
- Telemetry/Fusion regression: RMSE, MAE, NASA score (where available).
- Acoustic anomaly detection: ROC-AUC.
- Routing diagnostics: specialist split counts.

### 4.1.3 Provenance of Reported Numbers
All key numbers in this chapter come from executed notebook output cells and saved artifacts. Where a documented final report value differs from notebook-observed value, both are shown explicitly for traceability.

## 4.2 Telemetry Results

### 4.2.1 FD001 Global Attention-LSTM
FD001 delivered the strongest telemetry baseline in the project.

Observed output:
- RMSE: **14.6955**
- NASA Score: **307.5779**
- Accuracy (within +/- 15 cycles): **66.00%**

Interpretation:
- The model captures core degradation progression reliably for lower-complexity settings.
- Performance validates the sequence + attention design as a strong baseline architecture.

### 4.2.2 FD002 Global Attention-LSTM
Observed output:
- Final Test RMSE: **27.0426**

Interpretation:
- Error increase relative to FD001 is expected under added condition complexity.
- The result remains viable as a baseline but indicates sensitivity to distribution variation.

### 4.2.3 FD003: Global vs Fault-Specific MoE
Observed outputs:
- Global baseline RMSE: **21.04**
- MoE final RMSE: **20.2491**
- Router split: Specialist 0 = **57**, Specialist 1 = **43**

A separate recomputation from `fd003_moe_final_predictions.csv` confirms RMSE **20.2491** (100 rows).

Interpretation:
- FD003 benefits from specialist routing.
- Router-balanced distribution suggests meaningful subpopulation separation.
- Gain is modest but consistent with literature expectations for separable heterogeneity.

### 4.2.4 FD004: Global vs Asymmetric MoE
Observed outputs:
- Global baseline RMSE: **40.56** (precise print: 40.5655)
- MoE final RMSE: **41.5112**
- Router split: Specialist 0 = **185**, Specialist 1 = **63**

Interpretation:
- Unlike FD003, specialist decomposition did not improve RMSE on FD004.
- Asymmetric split indicates severe heterogeneity and potential router burden.
- Even without RMSE gain, specialist structure remains diagnostically useful for understanding behavior clusters.

### 4.2.5 Telemetry Summary Table

| Configuration | RMSE | Additional Details |
|---|---:|---|
| FD001 Global | 14.6955 | NASA score 307.5779; Accuracy +/-15 cycles 66.00% |
| FD002 Global | 27.0426 | Final test RMSE |
| FD003 Global | 21.04 | Baseline in MoE comparison |
| FD003 MoE | 20.2491 | Router split 57/43 |
| FD004 Global | 40.56 | Precise print 40.5655 |
| FD004 MoE | 41.5112 | Router split 185/63 |

## 4.3 Acoustic Results

### 4.3.1 Phase 1: Global Autoencoder Baseline
Observed output:
- ROC-AUC: **0.5906**

Interpretation:
- Baseline confirms model learns some normal manifold structure.
- Separability is weak, indicating overlap between normal and abnormal score distributions.

### 4.3.2 Phase 2: Sliding-Window Global Autoencoder
Observed output:
- ROC-AUC (Sliding Window): **0.5949**

Interpretation:
- Temporal granularity alone produced only marginal improvement.
- This supports the hypothesis that machine identity variability dominates anomaly signal in global settings.

### 4.3.3 Phase 3: Asset-Specific Sliding-Window AEs
Observed outputs:
- id_00: **0.4971**
- id_02: **0.5885**
- id_04: **0.5593**
- id_06: **0.8419**
- Mean AUC: **0.6217**

Interpretation:
- Asset-specific modeling improves average performance.
- Strong asset heterogeneity remains, with one asset near random-level separability (id_00).

### 4.3.4 Phase 4: Normalized Asset-Specific Final Model
Observed outputs:
- id_00: **0.6260**
- id_02: **0.9522**
- id_04: **0.7836**
- id_06: **0.9672**
- Mean AUC: **0.8323**

Interpretation:
- Per-asset normalization is the critical intervention.
- Large gains on id_02 and id_06 indicate substantial reduction in nuisance variance.
- id_00 remains comparatively weaker, highlighting intrinsic acoustic ambiguity or residual confounding.

### 4.3.5 Acoustic Comparative Tables

| Phase | Strategy | Mean ROC-AUC |
|---|---|---:|
| Phase 1 | Global AE | 0.5906 |
| Phase 2 | Sliding-window global AE | 0.5949 |
| Phase 3 | Asset-specific sliding-window AE | 0.6217 |
| Phase 4 | Normalized asset-specific AE | 0.8323 |

| Asset ID | Phase 3 | Phase 4 | Improvement |
|---|---:|---:|---:|
| id_00 | 0.4971 | 0.6260 | +0.1289 |
| id_02 | 0.5885 | 0.9522 | +0.3637 |
| id_04 | 0.5593 | 0.7836 | +0.2243 |
| id_06 | 0.8419 | 0.9672 | +0.1253 |
| **Mean** | **0.6217** | **0.8323** | **+0.2106** |

## 4.4 Multimodal Fusion Results

### 4.4.1 Notebook-Observed Fusion Run
From `multimodal_fusion_layer.ipynb`:
- Telemetry-only MoE RMSE: **41.5112**
- Multimodal fusion RMSE: **39.5560**

Absolute improvement:
\[
\Delta RMSE_{notebook} = 41.5112 - 39.5560 = 1.9552
\]

### 4.4.2 Documented Final Fusion Metrics
From `Documentation of PJT2 NASA.docx` final metric table:
- Telemetry-only RMSE: **41.51**
- Fusion RMSE: **39.13**
- Telemetry-only MAE: **32.12**
- Fusion MAE: **28.45**

Improvements:
\[
\Delta RMSE_{doc} = 41.51 - 39.13 = 2.38
\]
\[
\Delta MAE_{doc} = 32.12 - 28.45 = 3.67
\]

### 4.4.3 Why Two Fusion Rows Are Reported
Two rows are retained intentionally:
- notebook-observed metrics represent directly reproducible run output,
- documented-final metrics represent final selected model reported in project documentation.

This distinction avoids silent metric drift and improves reporting integrity.

### 4.4.4 Fusion Comparison Table

| Context | Telemetry RMSE | Fusion RMSE | Telemetry MAE | Fusion MAE |
|---|---:|---:|---:|---:|
| Notebook run | 41.5112 | 39.5560 | - | - |
| Documented final | 41.51 | 39.13 | 32.12 | 28.45 |

## 4.5 Explainability Outcomes (SHAP)

### 4.5.1 Global SHAP Behavior
The SHAP summary plot shows both fusion inputs contribute to final prediction. Contribution directions are consistent with model semantics:
- telemetry contributes baseline RUL structure,
- acoustic anomaly contributes corrective risk influence.

### 4.5.2 Maintenance Interpretation
A practical reading of SHAP behavior:
- when acoustic anomaly score is low, telemetry prediction remains dominant,
- when acoustic anomaly score is high, fused prediction is shifted downward (reduced RUL), acting as early warning override.

This is desirable in safety-sensitive maintenance contexts.

## 4.6 Visual Evidence

### 4.6.1 Acoustic Score Distribution
![Acoustic distribution](../images/acousticdataset_global_model_acoustic_anomaly_detection_autoencoder_model_ipynb_img01.png)

### 4.6.2 FD004 Global Baseline Plot
![FD004 global](../images/telemetrydataset_models_global_fd004_42_model_training_ipynb_img01.png)

### 4.6.3 FD004 Asymmetric MoE Plot
![FD004 MoE](../images/telemetrydataset_models_fault_specific_fd004_54_final_router_ipynb_img01.png)

### 4.6.4 Fusion Prediction Plot
![Fusion prediction](../images/multimodal_fusion_multimodal_fusion_layer_ipynb_img01.png)

### 4.6.5 SHAP Summary Plot
![SHAP summary](../images/multimodal_fusion_explainable_ai_shap_ipynb_img01.png)

## 4.7 Cross-Stage Discussion

### 4.7.1 What Improved Most
The largest subsystem gain came from acoustic Phase 4 normalization (+0.2106 mean AUC vs Phase 3), not from increasing model complexity alone.

### 4.7.2 Where Complexity Did Not Guarantee Gains
FD004 specialist MoE did not beat global RMSE, illustrating that decomposition is not universally beneficial. Routing complexity must be justified by robust separability and sufficient specialist data quality.

### 4.7.3 Why Fusion Still Helped
Even with mixed telemetry outcomes, fusion improved final RMSE versus telemetry-only baseline. This supports the hypothesis that modalities provide complementary failure evidence.

### 4.7.4 Practical Insight
A strong production strategy may be:
1. maintain a stable telemetry baseline,
2. strengthen acoustic asset-specific monitoring,
3. use interpretable fusion as a high-level decision layer.

## 4.8 Statistical and Validity Considerations

### 4.8.1 Internal Validity
Strengths:
- artifact-backed metrics,
- stage-wise evaluation,
- independent subsystem benchmarking.

Risks:
- potential run-to-run variance in notebook environments,
- metric divergence between notebook and documented final fusion run.

### 4.8.2 External Validity
The datasets are benchmark-grade but may not fully represent all industrial noise profiles, sensor faults, and domain shifts. Transfer performance should be validated before production adoption.

### 4.8.3 Construct Validity
RUL and anomaly metrics are standard, but mapping between anomaly score and maintenance action threshold remains deployment-specific.

## 4.9 Limitations
1. Fusion MAE not printed in available notebook output logs for the observed run.
2. Cross-modal alignment is decision-level, not raw synchronized multimodal sequence modeling.
3. Some notebook pipelines depend on path-sensitive local settings.
4. Specialist telemetry behavior on FD004 indicates unresolved routing/regime complexity.

## 4.10 Key Findings
1. FD001 global telemetry achieved strong benchmark-level performance.
2. FD003 benefited from specialist routing (21.04 -> 20.2491 RMSE).
3. FD004 did not gain RMSE from specialist MoE, highlighting complexity limits.
4. Acoustic subsystem became practically strong only after asset-specific normalization.
5. Fusion improved over telemetry-only baseline in both notebook-observed and documented-final evaluations.
6. SHAP interpretation aligned with intended maintenance logic.

## 4.11 Chapter Summary
This chapter demonstrated that subsystem-level rigor is essential in multimodal PHM. The project’s best outcomes came from targeted interventions (asset-aware normalization, interpretable late fusion) rather than complexity escalation alone. The final chapter consolidates these findings into conclusions, deployment insights, and future work priorities.

## 4.12 Quantitative Error Behavior Analysis

### 4.12.1 RMSE and MAE Interpretation in Maintenance Context
RMSE emphasizes larger errors, while MAE provides average absolute deviation. In maintenance planning:
- RMSE is useful for penalizing severe under/over-estimation events.
- MAE provides a stable estimate of typical prediction deviation.

Given prediction error \(e_i = \hat{y}_i - y_i\):

\[
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}e_i^2}, \quad MAE = \frac{1}{N}\sum_{i=1}^{N}|e_i|
\]

The documented fusion MAE improvement (32.12 -> 28.45) indicates not only fewer extreme misses but also better average-case prediction alignment.

### 4.12.2 NASA Score Implication
NASA score asymmetry penalizes late predictions more than early ones, reflecting safety prioritization in remaining-life forecasting. FD001 performance (NASA score 307.5779) indicates favorable behavior under this asymmetric risk profile.

## 4.13 Case-Oriented Interpretation of System Behavior

### 4.13.1 When Telemetry Dominates
In gradual degradation cases with stable acoustic conditions, telemetry predictions remain the primary determinant of fused output. This is expected and desirable because telemetry captures long-horizon trend information.

### 4.13.2 When Acoustics Corrects Telemetry
When mechanical anomalies become salient before clear telemetry divergence, acoustic risk signals can lower fused RUL estimates. This provides earlier cautionary behavior in mixed-signal scenarios.

### 4.13.3 Potential Conflict Cases
A challenging case occurs when telemetry suggests moderate remaining life but acoustic anomaly is high due to non-fault acoustic disturbances. This is where thresholding, confidence logic, and repeated observation windows become important for deployment.

## 4.14 Subsystem-by-Subsystem Strength and Weakness Profile

| Subsystem | Primary Strength | Primary Weakness |
|---|---|---|
| Telemetry global | Stable baseline on simpler subsets | Sensitive to complex regime/fault mixes |
| Telemetry specialists | Better local modeling for separable clusters | Router dependence and calibration overhead |
| Acoustic global | Simpler training workflow | Weak discrimination under identity confounding |
| Acoustic asset-specific normalized | Strong anomaly separability | Asset-wise variability still present |
| Fusion linear | Transparent and effective correction | Depends on quality of input feature alignment |

## 4.15 Why FD003 and FD004 Behaved Differently
FD003 and FD004 are often treated as “both hard,” but the experiments show qualitatively different behavior.

- FD003: specialist routing produced a clear improvement.
- FD004: specialist routing did not improve RMSE.

Possible reasons include:
1. FD004 has more severe overlap between latent regimes/fault signatures.
2. Asymmetric routing may create unequal specialist confidence profiles.
3. Error propagation from router assignment may offset specialist benefits.

This suggests future FD004 work should focus on adaptive/soft gating and confidence-aware routing rather than hard partitioning alone.

## 4.16 Acoustic Variability: Practical Significance
The acoustic subsystem shows that mean AUC can hide large per-asset spread.

Phase 4 AUC values:
- excellent for id_02 and id_06,
- moderate for id_04,
- comparatively weak for id_00.

Operational implication:
- deployment thresholds should be asset-specific,
- fleet-level KPIs should include per-asset reliability bands,
- one-size-fits-all anomaly thresholds are likely suboptimal.

## 4.17 Fusion Reliability Discussion

### 4.17.1 Why Linear Fusion Was Effective
Linear fusion provided gains despite simple form because it leveraged high-quality base signals and avoided overfitting under limited multimodal alignment complexity.

### 4.17.2 Coefficient Semantics and Risk Posture
Documented final fusion interpretation indicates telemetry as anchor and acoustic as negative-risk override. This is consistent with a conservative safety posture where strong anomaly evidence should reduce expected life estimates.

### 4.17.3 Calibration Considerations
For deployment, fusion calibration should include:
- coefficient stability checks across retrains,
- sensitivity analysis to acoustic score noise,
- periodic re-estimation under drift.

## 4.18 Result Consistency and Reporting Integrity
This report explicitly distinguishes:
- notebook-observed fusion RMSE (39.5560),
- documented final fusion RMSE/MAE (39.13 / 28.45).

This practice improves reporting integrity by preventing silent replacement of one run’s metrics with another. In real engineering teams, this distinction is crucial for reproducibility and model governance.

## 4.19 Threats to Validity (Expanded)

### 4.19.1 Data Validity
Benchmark datasets may not capture all production noise modes, sensor faults, or maintenance interventions.

### 4.19.2 Temporal Validity
The pipeline is evaluated offline. Real-time streaming and delayed-label settings can alter operational behavior.

### 4.19.3 Model Validity
Specialist and fusion performance may vary under retraining seeds, feature drift, and differing split conventions.

### 4.19.4 Interpretation Validity
SHAP explains model behavior, not causality. Decisions should combine model attribution with engineering diagnostics.

## 4.20 Deployment Readiness Checklist Based on Results
Before pilot deployment, the following checks are recommended:

1. Re-run full pipeline with fixed seeds and exported metric logs.
2. Validate per-asset acoustic thresholds on recent plant data.
3. Quantify false positive/negative tradeoff at maintenance decision threshold level.
4. Validate fusion behavior on known historical incidents.
5. Add confidence flags for low-quality or missing modality input.
6. Establish periodic recalibration policy.

## 4.21 Practical Takeaways for Engineering Teams

1. Do not assume model complexity equals reliability.
2. Start with traceable baselines, then add targeted complexity.
3. In acoustics, invest early in asset-aware normalization.
4. Treat fusion as a decision layer with explicit semantics.
5. Preserve run provenance and metric lineage.

## 4.22 Extended Chapter Summary
Chapter 4 confirms that the proposed multimodal framework is effective, but effectiveness is uneven across contexts and must be interpreted carefully. The best improvements emerged when methodological interventions matched the underlying failure mechanism of each modality. The chapter also demonstrated that transparent reporting of metric lineage (notebook vs documented final) strengthens technical credibility. These insights are consolidated into final conclusions and forward roadmap in Chapter 5.

## 4.23 Scenario Narratives for Maintenance Interpretation
To bridge model metrics and operational decisions, three representative scenario narratives are useful.

### Scenario A: Stable Operation, Low Risk
- Telemetry trend declines slowly and consistently.
- Acoustic anomaly score remains low.
- Fusion output remains close to telemetry baseline.

Operational action: continue monitoring, no urgent intervention.

### Scenario B: Emerging Mechanical Anomaly
- Telemetry remains near expected trend.
- Acoustic anomaly score spikes over recent windows.
- Fusion output drops relative to telemetry-only output.

Operational action: schedule targeted inspection earlier than telemetry-only policy would suggest.

### Scenario C: High Uncertainty Mismatch
- Telemetry indicates low remaining life.
- Acoustic score is inconsistent or noisy.
- Fusion result depends strongly on calibration and recent context.

Operational action: trigger engineer review with explanation trace before major maintenance action.

These narratives demonstrate why interpretable multimodal systems are operationally useful: they support different actions under different evidence configurations.

## 4.24 Result-to-Action Mapping Template

| Model Signal Pattern | Suggested Action |
|---|---|
| Low anomaly + moderate/high fused RUL | Routine monitoring |
| High anomaly + moderate fused RUL | Early inspection |
| High anomaly + low fused RUL | Priority maintenance planning |
| Conflicting signals + low confidence | Manual review and additional sensing |

This mapping is not a fixed policy but a decision-support scaffold that can be adapted per plant context.

## 4.25 Extended Discussion Closing
The results show that practical predictive maintenance quality is achieved through careful orchestration of subsystem assumptions, not through a single “best” model. Telemetry and acoustics contribute differently across scenarios, and fusion provides the mechanism to reconcile these differences at decision level. This systems view is the central empirical lesson of Chapter 4.
