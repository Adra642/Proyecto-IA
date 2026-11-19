---
doc_id: kaggle_data_dictionary
title: Data dictionary of the Bank Customer Churn dataset
source: kaggle_derived
source_type: dataset_stats
url: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn
language: en
geography: EU
topic: churn
columns:
- Age
- Balance
- Card Type
- Complain
- CreditScore
- EstimatedSalary
- Exited
- Gender
- Geography
- HasCrCard
- IsActiveMember
- NumOfProducts
- Point Earned
- Satisfaction Score
- Tenure
date: null
retrieved_at: '2026-11-19'
license: See Kaggle dataset page (license not verified)
format: generated
---

# Data dictionary: Kaggle "Bank Customer Churn" dataset

Personal identifiers (row number, customer id and surname) were removed before analysis and are not part of the knowledge base.

| Column | Type | Range or values | Meaning |
|---|---|---|---|
| CreditScore | integer | 350 to 850 | Credit score of the customer (higher is better). |
| Geography | category | France, Germany, Spain | Country of the customer: France, Germany or Spain. |
| Gender | category | Female, Male | Customer gender: Female or Male. |
| Age | integer | 18 to 92 | Age in years. |
| Tenure | integer | 0 to 10 | Years the customer has been with the bank. |
| Balance | decimal | 0 to 250,898.09 | Account balance (currency units of the dataset). |
| NumOfProducts | integer | 1 to 4 | Number of bank products held (1 to 4). |
| HasCrCard | integer | 0 to 1 | 1 if the customer has a credit card, otherwise 0. |
| IsActiveMember | integer | 0 to 1 | 1 if the customer is an active member, otherwise 0. |
| EstimatedSalary | decimal | 11.58 to 199,992.48 | Estimated annual salary. |
| Exited | integer | 0 to 1 | Target: 1 if the customer left the bank (churned), otherwise 0. |
| Complain | integer | 0 to 1 | 1 if the customer filed a complaint, otherwise 0. |
| Satisfaction Score | integer | 1 to 5 | Satisfaction score given by the customer, 1 (low) to 5 (high). |
| Card Type | category | DIAMOND, GOLD, PLATINUM, SILVER | Credit card tier: DIAMOND, GOLD, PLATINUM or SILVER. |
| Point Earned | integer | 119 to 1,000 | Loyalty points earned by the customer. |
