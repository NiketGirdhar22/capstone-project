# Chapter 2: Literature Review

## 2.1 Introduction to the Review Scope
This chapter synthesizes recent research (primarily 2023-2025) relevant to the capstone problem: multimodal predictive maintenance combining telemetry-based Remaining Useful Life (RUL) estimation, acoustic anomaly detection, fusion learning, and explainability.

The review is organized to mirror the project architecture:
1. telemetry/RUL modeling foundations,
2. acoustic anomaly modeling foundations,
3. multimodal fusion design patterns,
4. explainability methods for operational trust,
5. research gaps addressed by the proposed system.

The objective is not only to list papers but to extract design implications that directly informed the implementation choices in Chapters 3 and 4.

## 2.2 Predictive Maintenance: Conceptual Landscape
Modern PdM systems usually include four layers:
- sensing and data acquisition,
- health indicator extraction,
- prognosis/diagnosis modeling,
- decision support and maintenance action planning.

Deep learning has become dominant in the prognosis layer due to its ability to model nonlinear degradation. However, current literature consistently warns that raw predictive accuracy is insufficient for deployment. Reliability under operating-condition shifts, data imbalance, sensor heterogeneity, and interpretability constraints are now first-class concerns.

Recent reviews in PHM and fault diagnosis report three converging trends:
1. **hybrid temporal architectures** outperform static regressors on life-cycle tasks,
2. **transfer/domain adaptation** is essential under variable operating regimes,
3. **explainability and trustworthiness** increasingly determine whether models are adopted in maintenance workflows.

These trends strongly align with this project’s design: attention-enhanced sequence modeling, specialist routing for heterogeneous regimes, and SHAP-based interpretability for fused decisions.

## 2.3 Telemetry-Based RUL Estimation

### 2.3.1 CMAPSS as a Benchmark Family
NASA CMAPSS (FD001-FD004) remains the most referenced benchmark for turbofan degradation prognosis. The subsets vary in operating-condition and fault-mode complexity, enabling controlled comparison of model behavior under increasing realism. FD001 is typically easier (single fault mode, limited condition variability), while FD004 is substantially harder due to richer regime/fault interactions.

For this capstone, CMAPSS provides a strong testbed for both global and specialist telemetry strategies.

### 2.3.2 LSTM, Attention-LSTM, and Temporal Hybrids
A substantial body of work shows that recurrent temporal models (LSTM/GRU) remain competitive for RUL, especially when paired with attention modules. Attention mechanisms help by weighting critical degradation timesteps rather than treating all windows equally.

Key findings from recent RUL papers:
- attention-LSTM architectures improve long-horizon degradation tracking,
- CNN-LSTM hybrids extract local patterns before temporal integration,
- transformer/TCN variants may improve sequence context capture in some settings,
- gains often depend on preprocessing and operating-regime handling, not architecture alone.

This literature justifies the project’s use of attention-enhanced sequence regressors as the telemetry backbone.

### 2.3.3 Global Models vs Specialist Routing
Global models are simpler to train/deploy but can underfit heterogeneity when regimes and fault patterns vary strongly. Mixture-of-experts (MoE) and router-specialist systems are increasingly explored for this reason.

Recent MoE literature indicates:
- routing helps when latent subpopulations are separable,
- specialist gains are task-dependent and not guaranteed on all subsets,
- interpretability can improve because each specialist has a narrower failure context.

This motivated the FD003 and FD004 specialist tracks in this capstone.

### 2.3.4 Operating-Condition Shift and Domain Effects
Several 2023-2025 works emphasize that variable conditions distort degradation trajectories. Change-point-aware methods and domain adaptation are common countermeasures. The core lesson: a model that performs well in one regime can degrade sharply when condition distributions shift.

This risk connects directly to the project’s decision to compare global and fault-specific pipelines, and to maintain explicit preprocessing/routing artifacts.

## 2.4 Acoustic Anomaly Detection for Machinery

### 2.4.1 MIMII and Normal-Only Detection Reality
In industrial settings, abnormal labels are typically sparse, incomplete, or delayed. MIMII is widely used because it reflects this reality and supports normal-only anomaly learning paradigms.

Autoencoder-based detection remains practical because it avoids exhaustive fault labeling. The model learns normal manifolds; large reconstruction errors indicate out-of-distribution behavior.

### 2.4.2 Feature Representations and Temporal Resolution
Recent acoustic fault papers repeatedly show strong value in time-frequency representations (mel/log-mel/high-frequency features). Temporal modeling is essential because many anomalies are localized rather than persistent over full files.

However, temporal slicing alone does not solve machine identity confounding. If inter-machine acoustic signatures dominate reconstruction error, anomaly separability remains weak.

### 2.4.3 Asset-Specific Modeling and Normalization
A major conclusion in recent machinery acoustics literature is that asset-aware strategies materially improve reliability. Per-asset models and per-asset normalization reduce nuisance variance and improve anomaly contrast.

This directly informed the four-phase acoustic progression in this capstone:
1. global baseline,
2. sliding-window global,
3. sliding-window asset-specific,
4. normalized sliding-window asset-specific (final).

### 2.4.4 Evaluation Norms in Acoustic PHM
ROC-AUC remains the most common metric for anomaly discrimination under threshold-independent comparison. Yet many papers caution that mean AUC can hide per-asset variance. Asset-wise analysis is therefore critical for realistic deployment interpretation.

This is reflected in Chapter 4 where both mean and per-ID AUC values are reported.

## 2.5 Multimodal Learning for Condition Monitoring

### 2.5.1 Why Multimodal Fusion Matters
Single-modality PdM pipelines often fail in complementary ways:
- telemetry may miss early mechanical signatures,
- acoustics may overreact to non-health acoustic shifts.

Multimodal fusion aims to reduce these failure modes through complementary evidence integration.

### 2.5.2 Fusion Taxonomy
Common fusion paradigms:
- **Early fusion**: combine raw/low-level features before model learning.
- **Late fusion**: combine modality-specific model outputs.
- **Hybrid fusion**: combine intermediate and output-level signals.

For heterogeneous modalities with different scales and semantics, late/meta fusion is often more stable and interpretable.

### 2.5.3 Meta-Learner Fusion Findings
Recent multimodal fault-diagnosis papers report consistent gains from fusion over unimodal baselines. Practical lessons include:
- calibration between modalities is important,
- simple fusion models can outperform complex ones when data alignment is limited,
- linear/meta models improve auditability and deployment confidence.

These findings motivate the linear fusion meta-learner used in this project.

### 2.5.4 Trustworthy Fusion and PHM Requirements
The latest literature moves beyond “higher accuracy” toward trustworthy fusion: robustness, interpretability, and failure-aware behavior. This supports the project’s design choice to combine linear fusion with SHAP interpretation rather than relying on opaque nonlinear fusion only.

## 2.6 Explainability in Prognostics and Health Monitoring

### 2.6.1 Explainability as an Engineering Requirement
Maintenance teams need to know *why* a prediction changed, not just *what* it predicts. Without this, integrating AI into maintenance policy is difficult.

### 2.6.2 SHAP in PdM Context
SHAP is widely adopted because:
- it provides feature-level contribution values,
- it supports global and local interpretation,
- it can validate directionality (for example, rising anomaly risk should reduce predicted life).

Recent explainable PdM studies use SHAP/LIME/PDP/ICE combinations, but SHAP is especially practical for tabular fusion models.

### 2.6.3 Interpretation Risks and Good Practice
Literature also notes pitfalls:
- attribution does not guarantee causality,
- feature correlation can distort contribution narratives,
- explanation quality depends on baseline design and model stability.

This project treats SHAP as a decision-aid lens, not a causal proof.

## 2.7 Structured Review of Key Papers

| ID | Year | Theme | Representative Insight |
|---|---:|---|---|
| P1 | 2024 | Explainable PdM | SHAP/LIME/PDP/ICE improve maintenance interpretability. |
| P2 | 2025 | Acoustic anomaly | High-frequency + sequence models improve machine sound anomaly detection. |
| P3 | 2025 | Attention-LSTM RUL | Attention helps focus on degradation-relevant timesteps. |
| P4 | 2024 | RUL survey | Hybrid temporal architectures dominate modern RUL pipelines. |
| P5 | 2024 | Hierarchical transformer RUL | Attention hierarchies can improve complex temporal modeling. |
| P6 | 2024 | Change-point + RUL | Regime shifts require explicit adaptation. |
| P7 | 2024 | Domain adaptation RUL | Cross-condition robustness remains a major challenge. |
| P8 | 2024 | Multimodal fusion diagnosis | Sensor fusion outperforms unimodal diagnosis in many cases. |
| P9 | 2025 | Trustworthy multimodal fusion | Reliability and trust are becoming core fusion criteria. |
| P10 | 2025 | Interpretable ensemble RUL | Interpretability can directly support dynamic maintenance policy. |

## 2.8 Mapping Literature to Project Design Choices

| Literature Observation | Project Decision |
|---|---|
| Attention helps sequence prognosis | Attention-LSTM telemetry backbone |
| Regime/fault heterogeneity harms global models | FD003/FD004 specialist routing track |
| Normal-only acoustic learning is realistic | Autoencoder-based anomaly detection |
| Asset identity bias is strong in acoustics | Asset-specific and normalized acoustic phases |
| Late fusion is stable for heterogeneous modalities | Linear meta-learner fusion |
| Explainability is required for adoption | SHAP on fused model |

## 2.9 Research Gaps Addressed by This Capstone
The review reveals several persistent gaps:

1. **Limited RUL-oriented multimodal benchmarks** combining telemetry and acoustics.
2. **Insufficient asset-aware acoustic design** in many baseline studies.
3. **Sparse integration of explainability in end-to-end multimodal prognosis**.
4. **Few studies that explicitly compare global vs specialist routing and then fuse with orthogonal modality evidence**.

This capstone directly addresses these gaps by building a full, staged, and interpretable multimodal pipeline.

## 2.10 Critical Discussion and Positioning
This project is positioned as an engineering-focused multimodal PHM study rather than a pure architecture novelty paper. The contribution lies in the integration logic, stage-wise methodological rigor, and practical transparency.

The project’s strategic tradeoff is deliberate:
- use strong but manageable base models,
- prioritize traceable improvements over opaque complexity,
- keep the fusion layer interpretable for maintenance use.

This positioning makes the resulting system closer to deployment-oriented PHM workflows than many single-objective benchmark studies.

## 2.11 Chapter Summary
The literature supports the fundamental hypothesis behind this capstone: telemetry and acoustics are complementary, and their integration is most useful when condition heterogeneity and interpretability are handled explicitly. The next chapter translates this evidence into a complete methodology covering data pipelines, subsystem architectures, fusion formulation, evaluation equations, and reproducibility design.

## 2.12 References Used in This Chapter (Condensed)
- Gawde et al., IEEE Access, 2024, DOI: 10.1109/ACCESS.2024.3367110
- Le et al., IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3565812
- Dida et al., IJPHM, 2025, DOI: 10.36001/ijphm.2025.v16i2.4274
- Fan et al., Sensors, 2024, DOI: 10.3390/s24030824
- Wu et al., Sensors, 2024, DOI: 10.3390/s24113454
- Arunan et al., Control Engineering Practice, 2024, DOI: 10.1016/j.conengprac.2023.105840
- Du et al., Applied Soft Computing, 2024, DOI: 10.1016/j.asoc.2024.111717
- Zhou and Wang, RESS, 2024, DOI: 10.1016/j.ress.2024.110190
- Choudhary et al., IEEE TTE, 2025, DOI: 10.1109/TTE.2024.3502466
- Ying et al., Information Fusion, 2025, DOI: 10.1016/j.inffus.2025.103377

## 2.13 Comparative Methodological Critique
Recent works are strong in local innovation but often weak in system-level comparability. Three recurring critique points are especially relevant.

### 2.13.1 Inconsistent Evaluation Context
Different papers report on different subsets, split conventions, and preprocessing assumptions. This makes direct performance ranking difficult. A model can appear superior primarily because of evaluation context rather than true architectural advantage.

Implication for this capstone:
- prioritize internal consistency and stage-wise comparability over headline cross-paper ranking.

### 2.13.2 Underreporting of Failure Cases
Many studies emphasize average gains but under-report where methods fail (for example, specific assets, regimes, or fault modes). In predictive maintenance, these “failure slices” are often more operationally important than average behavior.

Implication for this capstone:
- include asset-wise acoustic performance and subset-wise telemetry behavior, even when results are mixed.

### 2.13.3 Explainability as a Postscript
A common pattern is to append explainability after model finalization without integrating it into decision semantics. This weakens practical value.

Implication for this capstone:
- use explainability directly in fusion interpretation and maintenance-oriented reasoning.

## 2.14 Telemetry Literature: Deeper Synthesis

### 2.14.1 Architecture vs Data Treatment
RUL literature frequently highlights architecture innovations (attention, transformers, CNN-LSTM hybrids), but meta-analysis across papers suggests data handling has comparable or larger impact:
- window construction policy,
- feature filtering/scaling,
- regime treatment,
- label construction details.

This supports an engineering-first approach where preprocessing and split discipline are treated as primary design dimensions.

### 2.14.2 Specialist Modeling in Context
MoE-style papers show gains when latent clusters are cleanly separable. However, when boundaries are ambiguous or router uncertainty is high, specialists can fragment training data and hurt global generalization.

This nuance is important: specialist architectures are not universally better; they are conditionally beneficial.

## 2.15 Acoustic Literature: Deeper Synthesis

### 2.15.1 The Identity-Health Entanglement Problem
A persistent issue in machine-sound anomaly detection is that models may encode identity/acoustic texture more strongly than health state. This causes unstable transfer and weak anomaly discrimination on some assets.

Asset-specific modeling and normalization are practical mitigation strategies repeatedly supported in recent studies.

### 2.15.2 Temporal Granularity Is Necessary but Not Sufficient
Sliding windows usually improve anomaly localization, but only modest gains appear if identity confounding is unresolved. This is why phase progression in the capstone separates temporal resolution changes from asset-aware normalization.

## 2.16 Fusion Literature: Deeper Synthesis

### 2.16.1 When Simple Fusion Wins
In data-limited or loosely aligned multimodal settings, linear or shallow fusion can outperform deeper fusion because:
- lower variance and better calibration,
- easier optimization,
- reduced risk of modality collapse.

This is especially relevant when one modality already dominates baseline variance (telemetry in this project).

### 2.16.2 Trust as a Performance Dimension
Recent trustworthy fusion studies argue that deployment readiness depends on more than RMSE/AUC:
- stable behavior under shift,
- interpretable contribution patterns,
- predictable failure modes.

This framing influenced the use of a transparent linear judge with SHAP analysis.

## 2.17 Explainability Literature: Deeper Synthesis

### 2.17.1 Attribution Utility in Maintenance
SHAP-style attributions are particularly useful when linked to actionable questions:
- Why did predicted life suddenly drop?
- Which signal changed most?
- Is this pattern consistent with known fault progression?

### 2.17.2 Attribution Governance
Literature increasingly recommends governance checks for explainability:
- stability across retraining,
- correlation awareness,
- consistency between local and global narratives.

This capstone uses SHAP as explanatory evidence, while acknowledging these constraints.

## 2.18 Literature-Derived Hypotheses for the Project
Based on the review, the project implicitly tested the following hypotheses:

1. **H1**: Attention-LSTM global telemetry models provide strong baseline prognosis on CMAPSS.
2. **H2**: Specialist routing helps subsets with meaningful latent separability (more likely in FD003 than FD004).
3. **H3**: Sliding-window acoustic modeling alone yields limited gains unless asset variability is controlled.
4. **H4**: Asset-specific normalization significantly improves acoustic anomaly discrimination.
5. **H5**: Interpretable late fusion improves over telemetry-only baseline in heterogeneous modality settings.

These hypotheses are validated or challenged in Chapter 4 with direct metrics.

## 2.19 Literature-to-Experiment Traceability Matrix

| Hypothesis Source in Literature | Experimental Check in This Project |
|---|---|
| Attention improves temporal degradation modeling | FD001-FD004 global Attention-LSTM runs |
| Specialist MoE can help heterogeneous subsets | FD003 and FD004 router-specialist comparisons |
| Global acoustic AEs are weak under identity variance | Phase 1 vs Phase 2 acoustic results |
| Asset-aware normalization is critical in acoustics | Phase 3 vs Phase 4 acoustic results |
| Late fusion improves robustness and interpretability | Telemetry-only vs fusion, plus SHAP |

## 2.20 Practical Lessons Extracted from Literature

1. Strong baselines matter more than novelty-only design.
2. Report subgroup behavior, not only aggregate means.
3. Treat preprocessing and normalization as modeling decisions.
4. Use explainability to support operations, not only publication narrative.
5. Preserve traceable artifact lineage for reproducibility.

These principles shaped the chapter structure of this report and the implementation style across the repository.

## 2.21 Extended Reference Pointers
Additional references from the curated project list that reinforce this chapter’s themes:
- Borst and Verhagen, The Aeronautical Journal, 2023, DOI: 10.1017/aer.2023.84
- Tian et al., Measurement, 2023, DOI: 10.1016/j.measurement.2023.112816
- Dong et al., MSSP, 2024, DOI: 10.1016/j.ymssp.2023.110900
- Guo et al., Journal of Big Data, 2024, DOI: 10.1186/s40537-024-01006-4
- Kumar et al., EAAI, 2023, DOI: 10.1016/j.engappai.2023.107126
- Li et al., IEEE Sensors Journal, 2023, DOI: 10.1109/JSEN.2023.3261874

## 2.22 Final Synthesis
The literature consistently supports a modular, multimodal, and explainable PdM architecture. It also warns against assuming that architecture complexity alone solves condition variability. The strongest evidence points toward disciplined preprocessing, context-aware modeling, and transparent fusion. These conclusions directly motivate the methodological choices detailed in Chapter 3.
