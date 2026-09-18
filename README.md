# Food Safety Inspection Analytics: Failure-Risk Prediction using German BVL Food Monitoring Data (2023)

## Project Overview

This project analyzes German food-monitoring data from the Bundesamt für Verbraucherschutz und Lebensmittelsicherheit (BVL) for 2023 to identify patterns associated with adverse food-safety findings and investigate whether machine-learning methods can support risk-based inspection prioritization.

The project combines data preparation, exploratory data analysis (EDA), statistical testing, machine learning, and Tableau visualization.

The predictive model is intended as a **screening and prioritization tool**. It is not intended to replace laboratory testing, regulatory assessment, or food-safety decisions.

---

## Business Problem

Food-monitoring authorities and inspection organizations work with limited inspection and laboratory capacity. Because adverse findings are rare, inspecting all samples with the same priority may not be the most efficient use of resources.

This project therefore investigates whether historical monitoring characteristics can help identify samples or product categories that may deserve greater inspection attention.

### Research Questions

1. Which food and product categories show higher observed rates of adverse findings?
2. Which sample and monitoring characteristics are associated with non-compliance?
3. Can characteristics available before the final laboratory outcome help predict elevated failure risk?

---

## Dataset

**Source:** German BVL Food Monitoring Data, 2023

The raw dataset contained approximately:

- **2.15 million analytical-result records**
- **9,711 original samples**
- **49 original columns**

Because multiple analytical-result records can belong to the same physical food sample, the raw data was transformed to a sample-level analytical dataset.

After restricting the analysis to food samples and completing the preparation process, the modelling dataset contained:

- **8,582 food samples**
- **71 adverse samples**
- **8,511 non-adverse samples**
- **0.83% adverse rate**

This represents a highly imbalanced classification problem.

---

## Target Definition

A conservative adverse target was constructed from BVL assessment information.

Examples of outcomes treated as adverse include:

- **Beanstandet** — Non-compliant / Objected
- **Standard nicht erfüllt** — Standard not met
- **> Höchstmenge (führt zur Beanstandung)** — Above maximum permitted level, leading to non-compliance

An observation recorded as above a maximum level **without** resulting in a formal non-compliance finding was not automatically classified as adverse.

---

## Data Preparation

Major preparation steps included:

1. Loading and inspecting the large raw BVL dataset.
2. Restricting the analytical scope to food products.
3. Aggregating analytical-result records to sample level.
4. Constructing the binary adverse/non-adverse target.
5. Engineering interpretable pre-outcome features.
6. Excluding identifiers and post-outcome information from model predictors to reduce target leakage.
7. Checking duplicates and missing values.
8. Creating modelling and Tableau-ready datasets.

The final modelling table contained **8,582 unique food samples**.

---

## Exploratory Data Analysis

The overall adverse rate was **0.83%**, but adverse findings were not distributed equally across product categories.

### Selected Product Groups

| Product Group | Observed Adverse Rate |
|---|---:|
| Gewürze (Spices) | 3.67% |
| Hülsenfrüchte, Ölsamen, Schalenobst (Legumes, Oilseeds and Nuts) | 1.63% |
| Fleisch warmblütiger Tiere, auch tiefgefroren (Meat, including frozen) | 1.43% |
| Frischobst einschließlich Rhabarber (Fresh Fruit including Rhubarb) | 0.99% |
| Overall Food Dataset | 0.83% |

### Selected Specific Products

| Specific Product | Adverse Rate | Adverse / Total |
|---|---:|---:|
| Sauerkirsche tiefgefroren (Frozen Sour Cherry) | 12.50% | 4 / 32 |
| Erdnuss geröstet mit Schale (Roasted Peanuts in Shell) | 3.64% | 8 / 220 |
| Süßkirsche (Sweet Cherry) | 3.33% | 6 / 180 |
| Pfeffer schwarz (Black Pepper) | 3.08% | 11 / 357 |

These are observed patterns in the 2023 monitoring dataset and should **not** be interpreted as proof that a product category is generally unsafe.

---

## Statistical Testing

A Chi-square test of independence was used to examine the relationship between **Product Group** and adverse-finding status.

Main result:

- **χ² ≈ 66.32**
- **p < 0.001**

The result provides statistical evidence of an association between Product Group and adverse-finding status in the dataset.

However, approximately **39.47% of expected cell counts were below five**, indicating sparse categories. The result was therefore interpreted cautiously.

Statistical association does not imply causation.

---

## Machine-Learning Methodology

The modelling dataset was divided using an **80/20 stratified train-test split** to preserve the rare adverse-class distribution.

### Split

| Dataset | Total Samples | Non-Adverse | Adverse |
|---|---:|---:|---:|
| Training | 6,865 | 6,808 | 57 |
| Test | 1,717 | 1,703 | 14 |

The test set remained untouched during model development.

Within the training data, **5-fold Stratified Cross-Validation** was used for model comparison and tuning.

Categorical predictors were transformed using one-hot encoding with handling for previously unseen categories.

### Models Compared

Four classifiers were investigated:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Because only 0.83% of observations were adverse, accuracy was not used as the main performance criterion.

A model predicting every observation as non-adverse would achieve approximately **99.17% accuracy while detecting zero adverse samples**.

The main evaluation metrics were therefore:

- Recall
- Precision
- F1 Score
- PR-AUC / Average Precision
- ROC-AUC as a secondary discrimination metric

Class weighting and SMOTE were investigated as imbalance-handling strategies. Class weighting proved more reliable for the final modelling workflow.

---

## Model Selection

After cross-validation and hyperparameter tuning, the tuned Random Forest provided the strongest cross-validated PR-AUC among the evaluated tuned models.

### Selected Cross-Validated Performance

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 3.80% | 44.24% | 6.99% | 5.46% | 77.81% |
| Tuned Random Forest | 4.30% | 42.12% | 7.78% | **10.32%** | **79.98%** |
| Tuned XGBoost | 6.79% | 14.39% | 9.22% | 6.42% | 75.57% |

The **Tuned Random Forest** was selected for final evaluation.

---

## Threshold Selection

The classification threshold was selected using **out-of-fold predictions from the training data**, rather than using the final test set.

A threshold of **0.40** was selected to improve adverse-case recall while maintaining a manageable screening volume.

The locked test set was used only after model development and threshold selection were completed.

---

## Final Test Results

The final Tuned Random Forest was evaluated on the untouched test set of **1,717 samples**, containing **14 adverse samples**.

### Final Performance

| Metric | Result |
|---|---:|
| Recall | **64.29%** |
| Precision | 2.98% |
| F1 Score | 5.70% |
| PR-AUC | **15.62%** |
| ROC-AUC | **82.11%** |
| Samples Flagged | **302 / 1,717 (17.59%)** |
| Adverse Samples Captured | **9 / 14** |

### Confusion Matrix

| | Predicted Non-Adverse | Predicted Higher Risk |
|---|---:|---:|
| Actual Non-Adverse | 1,410 | 293 |
| Actual Adverse | 5 | 9 |

The operational interpretation is:

> **By concentrating screening attention on approximately 17.6% of the test samples, the model captured 64.3% of the observed adverse samples.**

The low precision means that many flagged samples were false positives. The model should therefore be interpreted as a **risk-prioritization tool**, not an automatic food-safety classification system.

---

## Predictive Feature Importance

Grouped Random Forest feature importance indicated the following predictive contributions:

| Feature Group | Importance |
|---|---:|
| Specific Product | 27.71% |
| Product Group | 16.48% |
| Country | 11.12% |
| Sampling Month | 11.12% |
| Packaging | 10.61% |
| Business Type | 10.61% |
| Processing | 8.63% |
| Sampling Programme | 3.72% |

These values describe **predictive importance, not causal effects**.

For categorical variables, one-hot encoded feature importances were aggregated back to their original feature groups to improve interpretability.

---

## Tableau Dashboard

An interactive Tableau dashboard was developed to communicate the analytical findings and support exploration of the monitoring data.

### Dashboard Components

- Total Samples KPI
- Adverse Samples KPI
- Overall Adverse Rate
- Risk by Product Group
- Risk by Specific Product
- Monthly Risk Trend
- Machine-Learning Risk Prioritization Summary

### Live Dashboard

[View the Food Safety Inspection Analytics 2023 dashboard on Tableau Public](https://public.tableau.com/app/profile/paul.omogiate.obamwonyi/viz/Book1_17896449518250/Dashboard1?publish=yes)

The overall **0.83% adverse rate** is used as an analytical benchmark in the dashboard. It is **not a legal food-safety threshold**.

---

## Key Findings

1. Adverse findings represented only **0.83%** of food samples, creating a severe class-imbalance problem.
2. Adverse findings were concentrated more strongly in selected product groups and specific products.
3. Product characteristics provided meaningful predictive information.
4. Sampling Month also contributed predictive information, indicating that temporal patterns may contain useful signal.
5. The final Random Forest captured **9 of 14 adverse test samples** while flagging approximately **17.6% of test samples** for higher-risk screening.
6. Low precision means predictions should support prioritization rather than replace inspection or laboratory decisions.

---

## Recommendations

The model could potentially support risk-based monitoring by helping prioritize samples for additional inspection or laboratory attention.

Future development should include:

- Multi-year BVL monitoring data
- Temporal validation
- External or prospective validation
- Additional chemical and microbiological information where appropriate
- Probability calibration
- Cost-sensitive threshold optimization
- Investigation of more interpretable risk models

Laboratory testing and regulatory assessment should remain the final authority for food-safety decisions.

---

## Limitations

Important limitations include:

- Analysis based on one monitoring year: **2023**
- Only **71 adverse samples** in the final food dataset
- Only **14 adverse samples** in the test set
- Sparse observations in several categorical groups
- Low final-model precision
- No external validation
- No temporal validation
- Feature importance represents prediction, not causation

These limitations should be considered when interpreting the results.

---

## Repository Structure

```text
food-safety-inspection-analytics/
│
├── Data/
│   └── processed/
│
├── notebooks/
│   ├── data preparation / EDA notebooks
│   └── machine-learning notebook
│
├── outputs/
│   └── figures/
│
├── src/
│
├── config.py
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
