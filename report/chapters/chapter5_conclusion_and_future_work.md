# Chapter 5: Conclusion and Future Work

## 5.1 Conclusion
This capstone set out to design an end-to-end multimodal predictive maintenance framework that combines telemetry-based RUL estimation, acoustic anomaly detection, and interpretable decision fusion. The project objective was not only to improve a metric on one benchmark, but to build a structured system that reflects deployment realities: heterogeneous sensing, condition variability, and trust requirements.

Across the full pipeline, the project demonstrates five high-level outcomes.

1. **Telemetry subsystem maturity**: Attention-LSTM-based telemetry models delivered strong performance on easier CMAPSS subsets and useful baselines on harder subsets.
2. **Specialist routing value with caveats**: FD003 benefited from router-specialist decomposition, while FD004 showed that increased complexity does not automatically improve RMSE.
3. **Acoustic subsystem breakthrough via normalization**: The largest subsystem gain came from asset-specific normalized modeling, not from model complexity alone.
4. **Fusion benefit over telemetry-only baseline**: Both notebook-observed and documented-final evaluations showed fusion improvement.
5. **Interpretability alignment**: SHAP analysis provided directionally meaningful feature attribution consistent with maintenance reasoning.

Taken together, these outcomes support the central thesis of the project: **complementary modalities can improve predictive maintenance quality when integrated through a controlled, interpretable, and stage-wise methodology**.

## 5.2 Consolidated Contribution Summary

### 5.2.1 Technical Contributions
The capstone contributes a complete multimodal stack with explicit stage boundaries:
- telemetry preprocessing, sequence modeling, global and specialist tracks,
- acoustic feature engineering, four-phase anomaly subsystem evolution,
- fusion feature construction and linear meta-learning,
- SHAP-driven explanation layer.

### 5.2.2 Methodological Contributions
The work demonstrates a reproducible experimentation pattern:
- isolate subsystem behavior first,
- quantify incremental changes at each stage,
- integrate only after modality-level confidence is established.

### 5.2.3 Practical Contributions
The project provides practical guidance for maintenance AI teams:
- prioritize asset-aware normalization in acoustics,
- treat specialist routing as a conditional strategy,
- preserve interpretability in fusion decisions.

## 5.3 Deployment-Oriented Insights

### 5.3.1 Insight 1: Normalization is a First-Class Design Decision
In the acoustic branch, per-asset normalization produced the strongest improvement. This indicates that data conditioning can be more impactful than architecture changes when nuisance variance is high.

### 5.3.2 Insight 2: Specialist Models Should Be Evidence-Driven
Specialist routing improved FD003 but not FD004 RMSE. This suggests specialist decomposition should be applied when separability and router reliability are empirically supported, not assumed.

### 5.3.3 Insight 3: Interpretable Fusion is a Strong Baseline for Real Systems
Linear meta-fusion achieved measurable gains while preserving transparency. In industrial workflows where decisions are audited, this can be more valuable than marginally better but opaque alternatives.

### 5.3.4 Insight 4: Stage-Wise Traceability Reduces Reporting Risk
By preserving subsystem outputs and reporting both notebook-observed and documented-final fusion metrics, the project avoids ambiguity and makes model evolution auditable.

## 5.4 Limitations Revisited
No strong capstone should end without clarifying limits.

1. **Benchmark-constrained validation**: Results are based on CMAPSS and MIMII; direct transfer to every industrial context is not guaranteed.
2. **Alignment granularity**: Fusion is performed at decision level rather than synchronized raw multimodal sequence level.
3. **Run variability**: Notebook and documented final fusion metrics differ, indicating separate final-run calibration.
4. **FD004 telemetry complexity**: Specialist routing did not improve RMSE in the hardest subset.

These limitations do not invalidate the findings, but they define clear boundaries for interpretation.

## 5.5 Future Work Directions

## 5.5.1 Direction A: Time-Synchronized Multimodal Modeling
Current fusion operates on subsystem outputs. A next step is tightly synchronized multimodal sequence learning where telemetry and acoustic windows are jointly modeled over aligned time spans.

Potential approaches:
- cross-attention between telemetry and acoustic embeddings,
- temporal alignment layers with lag compensation,
- uncertainty-aware multimodal transformers.

## 5.5.2 Direction B: Uncertainty-Aware Prognostics
Maintenance decisions are risk-sensitive. Beyond point estimates, future models should produce calibrated uncertainty intervals:

\[
\hat{y} \pm \sigma_{pred}
\]

This enables confidence-aware scheduling and policy thresholds.

## 5.5.3 Direction C: Adaptive Routing and Gating
For high-complexity subsets like FD004, future work can explore:
- soft gating instead of hard routing,
- hierarchical experts with regime-conditioned priors,
- dynamic ensemble weighting based on confidence or drift signals.

## 5.5.4 Direction D: Robustness Under Domain Shift
Real deployments encounter shift in noise, environment, and load profile. Future validation should include:
- out-of-domain tests,
- transfer adaptation studies,
- continual calibration under streaming data.

## 5.5.5 Direction E: Maintenance Policy Coupling
A valuable extension is integrating prediction outputs with explicit maintenance optimization:
- cost-aware objective functions,
- downtime-risk tradeoff simulation,
- threshold recommendation under service constraints.

## 5.5.6 Direction F: Stronger Explainability Protocols
SHAP is useful, but future explainability can include:
- local case narratives for operator dashboards,
- counterfactual analysis,
- explanation stability checks across retraining cycles.

## 5.6 Recommended Next Iteration Plan
A practical roadmap for the next project phase:

1. Standardize all notebook paths and package reproducible scripts.
2. Re-run full pipeline with fixed seeds and metric export logs.
3. Add synchronized multimodal dataset mapping layer.
4. Evaluate linear vs calibrated nonlinear fusion under the same split.
5. Add uncertainty intervals and threshold analysis for maintenance decisions.
6. Build a lightweight dashboard with prediction + SHAP traces.

## 5.7 Final Closing Statement
The project demonstrates that multimodal predictive maintenance is most effective when treated as a systems-engineering problem, not just a model-selection problem. Telemetry and acoustics each contribute distinct health evidence. Their careful integration, combined with transparent interpretation, produces a more dependable predictive signal than either modality alone.

In summary, this capstone provides a strong foundation for a deployable and explainable predictive maintenance platform, while clearly identifying what must be improved for industrial-grade operationalization.

## 5.8 Deployment KPI Framework
A practical next step is to evaluate the system not only by model metrics but also by maintenance KPIs.

Recommended KPI categories:

1. **Prediction Quality KPIs**
- RMSE/MAE on rolling validation sets,
- anomaly precision/recall at operational thresholds,
- calibration drift indicators.

2. **Maintenance Outcome KPIs**
- reduction in unplanned downtime,
- increase in average warning lead-time,
- maintenance intervention success rate.

3. **Operational Trust KPIs**
- percentage of alerts with interpretable attribution,
- engineer acceptance rate of model recommendations,
- number of false-critical alerts per month.

A KPI-centered evaluation closes the gap between benchmark success and plant-floor value.

## 5.9 Governance and Lifecycle Recommendations
A production lifecycle should include:

- **Model Registry Discipline**: version all telemetry, acoustic, and fusion artifacts.
- **Retraining Protocols**: trigger retraining on drift or periodic cadence.
- **Audit Trails**: preserve input features, outputs, and explanations for critical alerts.
- **Fallback Policies**: define safe behavior when one modality is unavailable.

This governance layer is essential for long-term reliability and compliance.

## 5.10 Suggested Technical Backlog for Next Semester

1. Implement automated evaluation scripts that export unified CSV metric reports.
2. Add uncertainty estimation to both telemetry and fusion layers.
3. Add soft-gated MoE routing experiments for FD004-like complexity.
4. Test robust fusion alternatives (Huber regression, quantile regression).
5. Build an operator-facing interface with case-level trend and SHAP view.

## 5.11 Final Reflection
The most important lesson from this capstone is that predictive maintenance performance depends on *alignment between method and signal behavior*. The strongest gains came from choices that addressed real signal structure:
- routing where separability exists,
- normalization where identity confounding dominates,
- fusion where modality evidence is complementary,
- interpretation where operational trust is required.

This is why the project should be viewed as a systems design contribution rather than a single-model contribution. The resulting framework is modular, explainable, and extensible, making it a strong base for future industrial pilot work.

## 5.12 Practical Adoption Roadmap (12-Month View)
A realistic adoption roadmap can be staged as follows:

### Phase 1 (Month 1-3): Stabilization
- lock reproducible training/evaluation scripts,
- finalize metric export format,
- run baseline replay with fixed seeds.

### Phase 2 (Month 4-6): Shadow Deployment
- connect offline pipeline to mirrored production data,
- run alerts in non-action mode,
- collect engineer feedback and false-alert diagnostics.

### Phase 3 (Month 7-9): Controlled Actioning
- enable limited maintenance recommendations,
- track intervention outcomes,
- tune thresholds and confidence gates.

### Phase 4 (Month 10-12): Scale and Governance
- formalize retraining cadence,
- add drift alarms and audit dashboards,
- standardize model card updates for each release.

This phased rollout reduces operational risk and helps maintain trust while the system matures.

## 5.13 Action Checklist for Immediate Next Steps
If this report is used as a handover baseline, the following immediate steps are recommended:

1. Freeze current artifacts and create a tagged snapshot.
2. Export all chapter tables to a central metrics sheet.
3. Validate image references and figure numbering for final submission formatting.
4. Execute one clean reproducibility run and attach generated logs.
5. Document any metric differences between rerun and this report.

This checklist ensures continuity between capstone completion and future extension work.

## 5.14 Closing Note
The value of this project is not only in the reported metrics, but in the reusable structure it leaves behind. Future teams can iterate subsystem-wise, verify improvements with clear baselines, and maintain interpretability while scaling complexity. That continuity is a major success criterion for capstone-to-production transitions.
