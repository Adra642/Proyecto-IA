---
doc_id: kaggle_findings
title: 'Findings: churn drivers, Complain leakage and satisfaction'
source: kaggle_derived
source_type: dataset_stats
url: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
language: en
geography: EU
topic: churn
columns:
- Age
- Complain
- Exited
- Geography
- IsActiveMember
- NumOfProducts
- Satisfaction Score
date: null
retrieved_at: '2026-11-19'
license: See Kaggle dataset page (license not verified)
format: generated
---

# Findings: churn drivers in the Kaggle "Bank Customer Churn" dataset

Overall churn rate: 20.4%.

## Top churn drivers (excluding Complain, see the leakage check)

By absolute correlation with Exited:

1. Age (correlation +0.285)
2. IsActiveMember (correlation -0.156)
3. Balance (correlation +0.119)
4. Gender (correlation -0.106)
5. NumOfProducts (correlation -0.048)

By mutual information with Exited:

1. NumOfProducts (mutual information 0.0698)
2. Age (mutual information 0.0667)
3. Geography (mutual information 0.0140)
4. IsActiveMember (mutual information 0.0123)
5. Gender (mutual information 0.0056)

## Leakage check: Complain versus Exited

- Churn rate among customers who complained: 99.5%
- Churn rate among customers who did not complain: 0.1%
- Share of churned customers who had complained: 99.8%
- Correlation between Complain and Exited: 0.996

Conclusion: Complain almost perfectly mirrors Exited, so it is very likely target leakage (the complaint is probably recorded as part of the departure). A churn model should not use Complain as a predictor.

## Satisfaction score versus churn

Churn rate ranges from 19.6% to 21.8% across the five satisfaction scores (correlation with Exited: -0.006).

Conclusion: satisfaction score alone does not track churn in this dataset; the churn rate is almost flat across scores.
