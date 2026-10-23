# Model Evaluation Report

**Model:** Logistic Regression (`class_weight="balanced"`, L2 regularisation, C=1.0)
**Data:** IBM Telco Customer Churn - 7,043 customers, 26.5% churned
**Split:** stratified 80/20 (train 5,634 / test 1,409), random_state=42

## 1. Headline results (test set, threshold 0.50)

| Metric | Value |
|---|---|
| Accuracy | 0.740 |
| Precision (churn class) | 0.507 |
| Recall (churn class) | 0.786 |
| F1-score (churn class) | 0.616 |
| ROC-AUC | 0.842 |

Confusion matrix: TN=749, FP=286, FN=80, TP=294.

- **Precision** = TP / (TP + FP): of the customers flagged as churners, 51% really churned.
- **Recall** = TP / (TP + FN): of the customers who really churned, the model caught 79%.
- **F1** is the harmonic mean of precision and recall.

## 2. Why accuracy alone is misleading

Only 26.5% of customers churn. A "model" that predicts *nobody churns* would score
**73.5% accuracy** while catching **0%** of churners (recall = 0) - useless for retention.
Our model has *lower* accuracy (74.0%) but finds 79% of the customers who leave,
which is what the business actually needs. For imbalanced problems, look at precision, recall and F1 for the minority class.

## 3. Threshold trade-off

Lowering the threshold catches more churners (higher recall) but flags more loyal customers by mistake (lower precision).
The business should choose the threshold from the relative cost of a retention offer vs. a lost customer.

| threshold | accuracy | precision | recall | f1 |
|---|---|---|---|---|
| 0.3 | 0.656 | 0.432 | 0.928 | 0.589 |
| 0.4 | 0.7 | 0.466 | 0.866 | 0.606 |
| 0.5 | 0.74 | 0.507 | 0.786 | 0.616 |
| 0.6 | 0.762 | 0.54 | 0.703 | 0.611 |
| 0.7 | 0.79 | 0.604 | 0.604 | 0.604 |

## 4. Risk categories

Bands use out-of-fold (5-fold cross-validated) probabilities: Low < 0.30, Medium 0.30-0.60, High >= 0.60.
The observed churn rate rises steadily across the bands, which suggests the bands are meaningful.
Note: because the model is class-balanced, the scores rank customers well but run higher than the real churn rates,
so read them as a risk score; the "actual churn rate" column below is the true observed rate.

| risk band | customers | actual_churn_rate |
|---|---|---|
| Low | 2966 | 4.8% |
| Medium | 1722 | 23.5% |
| High | 2355 | 56.2% |

## 5. Factors most associated with churn

Coefficients are on the log-odds scale; odds ratio > 1 means higher churn odds. Numeric features are standardised (per 1 std-dev).

| feature | coefficient | odds_ratio | direction |
|---|---|---|---|
| tenure | -1.154 | 0.315 | lowers churn risk |
| InternetService = No | -1.133 | 0.322 | lowers churn risk |
| InternetService = Fiber optic | 1.042 | 2.834 | raises churn risk |
| Contract = Two year | -0.755 | 0.47 | lowers churn risk |
| Contract = Month-to-month | 0.682 | 1.978 | raises churn risk |
| TotalCharges | 0.486 | 1.626 | raises churn risk |
| MonthlyCharges | -0.366 | 0.694 | lowers churn risk |
| PaymentMethod = Electronic check | 0.259 | 1.296 | raises churn risk |

## 6. Model comparison (5-fold cross-validation)

Five models were compared on identical folds. The tuned model searches C in {0.01, 0.1, 1, 10} with an inner 3-fold grid search (nested CV).
Engineered features: number of services, average monthly spend, current bill vs. average, auto-pay flag, new-customer flag.

| model | roc_auc | pr_auc | precision | recall | f1 | roc_auc_std |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.845 | 0.656 | 0.513 | 0.801 | 0.626 | 0.013 |
| Logistic Regression + engineered features | 0.847 | 0.661 | 0.52 | 0.8 | 0.63 | 0.011 |
| Logistic Regression + engineered, tuned C | 0.847 | 0.661 | 0.521 | 0.8 | 0.631 | 0.012 |
| Random Forest | 0.844 | 0.654 | 0.559 | 0.706 | 0.624 | 0.01 |
| Gradient Boosting | 0.844 | 0.656 | 0.517 | 0.796 | 0.627 | 0.009 |

**Finding:** all models land within about 0.003 ROC-AUC of each other. The extra complexity of Random Forest and Gradient Boosting buys no
meaningful accuracy here, so the simpler and explainable Logistic Regression is a justified choice.

## 7. Business value: retention-campaign ROI

Assumptions (editable in the dashboard): $50 cost per retention offer, 30% of would-be churners
saved by an offer, each saved customer retained for 12 months of billing.

Best threshold on cross-validated scores: **0.45** -> target 3,190 customers,
reach 1,563 real churners, campaign cost $159,500,
revenue saved $432,266, **net benefit $272,766** (ROI 1.7x).
The best threshold is a business choice that depends on these costs, not a fixed 0.50.
This is an illustrative back-test: the threshold is chosen and scored on the same data.

## 8. Limitations

- **Associations, not causes.** The model shows which customer attributes go with churn; it does not prove *why* a customer leaves.
- Single dataset snapshot; no time dimension, so performance on future customers may differ. For churned customers, `tenure` is their final lifetime, so it partly reflects the outcome.
- `TotalCharges` is strongly correlated with `tenure` x `MonthlyCharges`, so individual coefficients for those three features should be read together, not in isolation (multicollinearity).
- Logistic Regression is linear; interactions (e.g. contract x tenure) are not modelled. Tree-based models could be tried next.
- Retention actions in the dashboard are rule-based suggestions and have not been A/B tested.
