# Literature gold answers: verification sheet

The gold answers of L1 and L2 are **drafts written from the arXiv abstracts**. Please check each against the paper (link below) and tick the box. Sentences are quoted from the downloaded PDF with their page number; the full text is not republished.

## L1
- Paper: https://arxiv.org/abs/2301.13099 (PDF: https://arxiv.org/pdf/2301.13099)
- Question: In the arXiv study 'Prediction of Customer Churn in Banking Industry', which classifier did the authors find to be the best performing, and what structure did it have?
- Draft gold answer: DRAFT, human must verify against the PDF: the study compares six supervised classifiers on 10 attributes of 10,000 European bank customers and concludes an ANN with five nodes in a single hidden layer performs best.
- Supporting sentences found in the PDF:
  - p.1: "This study compares the performance of six supervised classification techniques to suggest an efficient model to predict customer churn in banking industry, given 10 demographic and personal attributes from 10000 customers of European banks."
  - p.1: "The effect of feature selection, class imbalance, and outliers will be discussed for ANN and random forest as the two competing models."
  - p.1: "As shown, unlike random forest, ANN doesn’t reveal any serious concern regarding overfitting and is also robust to noise."
  - p.1: "Therefore, ANN structure with five nodes in a single hidden layer is recognized as the best performing classifier."
  - p.7: "On the other hand, when employing decision trees and random forest, none of these further pre-processing steps are required."
  - p.7: "Hyperparameter tuning is another step which makes the modeling step by k-NN, decision trees, random forest and ANN time-consuming."
  - p.7: "Following the implementation of random Forest and ANN, and achieving larger values of kappa (0.5 < kappa < 0.6) and F1 the model improves to some extent."
  - p.7: "As appears, overfitting is substantial for random forest while models developed upon Naïve Bayes and decision trees are slightly underfitted."
- [ ] Verified by a human against the PDF (initials/date: ______ )

## L2
- Paper: https://arxiv.org/abs/2606.07582 (PDF: https://arxiv.org/pdf/2606.07582)
- Question: Which model architecture does the arXiv paper on customer churn prediction on structured data with FT-Transformer and stacking ensembles propose, and what problems of earlier research does it say it addresses?
- Draft gold answer: DRAFT, human must verify against the PDF: a hybrid of feature-tokenized transformers (FT-Transformer) with gradient-boosted trees through calibration-aware stacking, addressing gaps in statistical validation, probability calibration and reproducibility.
- Supporting sentences found in the PDF:
  - p.1: "This study introduces a validated hybrid architecture that integrates feature- tokenized transformers (FT-Transformer) with gradient-boosted trees through calibration-aware stacking."
  - p.1: "The proposed framework addresses persistent gaps in statistical validation, probability calibration, and reproducibility found in prior research."
  - p.1: "The FT-Transformer captures higher-order feature interactions using self-attention, while XGBoost captures gradient-boosted decision boundaries with complementary inductive biases."
  - p.1: "The models are ensembled using out-of-fold (OOF) stacking with a logistic regression meta-learner, which recalibrates overconfident base model outputs and learns optimal combination weights."
  - p.1: "Ablation studies demonstrate that both the transformer component and stacking strategy contribute materially to performance."
  - p.1: "The proposed methodology offers a reproducible and extensible reference architecture for contemporary churn prediction on structured tabular data, bridging recent advances in attention-based modeling with ensemble techniques."
  - p.1: "INDEX TERMS Customer churn prediction, FT-Transformer, gradient boosting, stacking ensemble, tabular data, class imbalance, probability calibration, reproducible machine learning I."
  - p.2: "Roy et al.: Customer Churn Prediction Using FT-Transformer and Stacking Ensembles policyholders and an $800 average CLV would preserve around $800,000 annually."
- [ ] Verified by a human against the PDF (initials/date: ______ )

