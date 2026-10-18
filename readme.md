# Predicting Insurance Claim Filing - Capstone Project

## 1. Problem Statement

**Research question:** Will a policyholder file an auto insurance claim during the policy period?

Insurers need to estimate which policies are likely to produce a claim in order to price risk, set reserves and focus loss-prevention efforts. This project builds a binary classification model that predicts `claim_status` (1 = claim filed, 0 = no claim) from the policy's tenure, the policyholder's age and region, and the insured vehicle's attributes.

## 2. Data

**Source:** *Insurance Claims Dataset* by Sergey Litvinenko (Kaggle), stored at [data/Claimsdata.csv](data/Claimsdata.csv).

- **58,592 policies and 41 columns**, with no missing values.
- **Target:** `claim_status`, which is heavily imbalanced: 93.6% no claim (54,844) and 6.4% claim (3,748), a ratio of about 14.6 : 1.

| Group | Columns |
|---|---|
| Identifier | `policy_id` |
| Policy | `subscription_length` (policy tenure) |
| Policyholder | `customer_age`, `region_code`, `region_density` |
| Vehicle | `vehicle_age`, `model`, `segment`, `fuel_type`, `engine_type`, `max_torque`, `max_power`, `displacement`, `cylinder`, `transmission_type`, `steering_type`, `rear_brakes_type`, `turning_radius`, `length`, `width`, `gross_weight`, `airbags`, `ncap_rating` |
| Vehicle safety and comfort flags (Yes/No) | 17 `is_*` columns, such as `is_esc`, `is_tpms`, `is_parking_camera`, `is_brake_assist`, `is_speed_alert` |

The dataset does **not** contain gender, occupation, credit score, premium, deductible or prior-claim history, so those factors are outside the scope of this analysis.

## 3. Techniques Applied

### 3.1 Exploratory Data Analysis (EDA)

- Target distribution (class imbalance).
- Categorical value counts and a redundancy check of the vehicle columns against `model`.
- Distributions, skewness and outliers of the numeric features.
- Numeric features vs. target (KDE and box plots) and claim rate by category (bar charts against the overall 6.4% claim rate).
- Correlation heatmap and identification of highly correlated pairs.

### 3.2 Data Preprocessing and Feature Engineering

- Converted the `is_*` Yes/No flags to 1/0.
- Parsed `max_torque` and `max_power` into numbers and derived `power_to_weight` (bhp / gross weight).
- Built a `safety_score` from the safety flags, `airbags` and `ncap_rating`.
- Capped `vehicle_age` at the 99th percentile (4.2 years).
- Dropped `policy_id`, `region_density` (it duplicates `region_code`), `model` and the raw vehicle columns. This reduced the dataset from 41 to 10 columns.
- Final features: `subscription_length`, `vehicle_age`, `customer_age`, `safety_score`, `power_to_weight`, `region_code`, `segment`, `fuel_type`, `transmission_type`.
- Screened feature importance with mutual information and a Random Forest.
- Used a stratified 80/20 train/test split, with a `ColumnTransformer` that scales numeric features and one-hot encodes categorical features.

### 3.3 Predictive Modeling Techniques

- **Baseline:** a `DummyClassifier` that always predicts "No Claim". It shows why accuracy is misleading on this data.
- **Models:** class-balanced logistic regression and a class-balanced Random Forest, each compared with the dummy baseline.
- **Cross-validation and tuning:** 3-fold stratified cross-validation on the training split. `GridSearchCV` selects each model's hyperparameters using average precision (PR-AUC); preprocessing is inside the pipeline and is fit independently within each fold. Logistic regression searches `C` in `[0.01, 0.1, 1, 10]`. Random Forest searches `max_depth` in `[8, 16]` and `min_samples_leaf` in `[5, 20]` with 100 trees.
- **Evaluation metrics:** precision, recall, F1, ROC-AUC and PR-AUC on the held-out test set. PR-AUC is the tuning metric because the positive class is rare; the no-skill PR-AUC reference is approximately 0.064.

## 4. Key Findings

1. **The data is highly imbalanced.** A model that always predicts "No Claim" is 93.6% accurate but catches no claims.
2. **The vehicle features are largely redundant.** All 33 vehicle columns have exactly one value per `model`, and there are only 11 models. The size and engine columns are highly correlated (r = 0.81–0.96). In practice the data has only five independent sources of information: tenure, vehicle age, customer age, region and vehicle model.
3. **Policy tenure is the strongest predictor.** Claim rate rises from 3.8% for tenure of 2 or less to about 9% for tenure of 8–12. Median tenure is 8.3 for policies with a claim vs 5.6 for policies without. Tenure ranks first in both mutual information and Random Forest importance.
4. **Region matters.** Claim rates range from about 3.9% to about 10.8% across the 22 regions.
5. **Other factors have weak effects.** Policies with claims have slightly newer vehicles (median 1.0 vs 1.2 years), and older customers (55+) claim slightly more often. Vehicle model shifts the claim rate only between 4.1% and 7.4%. Segment, fuel type, NCAP rating and airbags stay close to the 6.4% average, and fuel type and transmission type rank lowest in feature importance.
6. **Linear signal is weak.** No numeric feature has an absolute correlation with `claim_status` above 0.08.

**Cross-validation results:**

| Model | Mean CV PR-AUC | Standard deviation |
|---|---:|---:|
| Logistic Regression (best `C=0.01`) | 0.089 | 0.004 |
| Random Forest (100 trees, depth 8, leaf size 20) | 0.105 | 0.003 |

Random Forest was selected as the stronger model family by mean cross-validated PR-AUC. Hyperparameters were selected using only the training split.

**Held-out test results (11,719 policies):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.936 | 0.000 | 0.000 | 0.000 | 0.500 | 0.064 |
| Tuned Logistic Regression | 0.571 | 0.087 | 0.604 | 0.153 | 0.624 | 0.093 |
| Tuned Random Forest | 0.587 | 0.098 | 0.664 | 0.171 | 0.665 | 0.106 |

The tuned Random Forest improves PR-AUC over the 0.064 no-skill reference and finds about two-thirds of claims at the default threshold. Precision is still low: most policies it flags do not have a claim. The available features provide only modest predictive separation, so the model should not be used for automated pricing or adverse policy decisions.

**Business recommendations and next steps:** consider a controlled loss-prevention outreach pilot, not automated underwriting. Work with claims and pricing teams to set a decision threshold based on the relative cost of missed claims and unnecessary outreach. Validate on a later time period, monitor precision, recall and calibration, and investigate additional permissible predictors such as prior claims and coverage details.

## 5. Repository Structure

```
ML_Project_Captstone_Module24/
├── capstoneProject.ipynb      # Main notebook: EDA, feature engineering, cross-validated tuning
├── data/
│   └── Claimsdata.csv         # Insurance claims dataset (58,592 rows)
└── readme.md
```

## 6. How to Run the Project

**Prerequisites:** Python 3.9 or later and VS Code with the Python and Jupyter extensions (or JupyterLab).

1. Clone or download this folder and open it in VS Code.
2. Create and activate a virtual environment from the project folder:

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn ipykernel
   ```
   The project was tested with Python 3.9.6, pandas 2.3.3, numpy 2.0.2, matplotlib 3.9.4, seaborn 0.13.2 and scikit-learn 1.6.1.
4. Open [capstoneProject.ipynb](capstoneProject.ipynb), select the `.venv` kernel and choose **Run All**. The notebook reads `data/Claimsdata.csv` with a relative path, so the working directory must be the project folder.

