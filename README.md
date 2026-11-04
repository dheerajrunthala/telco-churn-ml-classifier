# Telco Customer Churn - Binary Classification

**M.Tech ML Assignment 2 | BITS Pilani**
Author: Dheeraj Runthala (2026AJ05331)

Five classification models trained on the IBM Telco Customer Churn dataset,
compared on six metrics, and served through a Streamlit app that scores any
uploaded test CSV.

---

## a. Problem statement

A telecom provider loses revenue every time a subscriber leaves, and winning a
customer back costs far more than keeping one. The task is to predict, from a
customer's account profile and the services they subscribe to, whether they will
churn.

This is a **binary classification** problem. The target `Churn` takes the value
`Yes` or `No`, encoded as `1` and `0`, with churn as the positive class.

The dataset is imbalanced: only 26.54% of customers churned. That shapes how the
results must be read. A model that predicts "nobody churns" scores 73.46%
accuracy while being useless, so accuracy alone cannot rank these models.
Recall, F1 and MCC are reported alongside it, and MCC is treated as the headline
figure because it only scores high when both classes are predicted well.

---

## b. Dataset description

| | |
|---|---|
| **Source** | IBM Sample Data Sets - Telco Customer Churn ([Kaggle mirror](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| **File** | `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Rows** | 7,043 customers |
| **Columns** | 21 (1 identifier + 19 features + 1 target) |
| **Target** | `Churn` (`Yes` / `No`) |

### Features

**3 numeric:** `tenure` (months with the company, 0-72), `MonthlyCharges`,
`TotalCharges` (19-8685).

**16 categorical:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`,
`PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
`OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`,
`StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

After one hot encoding, the 19 raw features expand to **46 columns**.

### Class distribution

| Class | Count | Share |
|---|---|---|
| No (retained) | 5,174 | 73.46% |
| Yes (churned) | 1,869 | 26.54% |

### Cleaning performed

1. **Dropped `customerID`** - a unique identifier with no predictive signal or anything.
2. **`TotalCharges` converted to numeric.** It ships as text because 11 rows
   hold a blank string instead of a value (all are customers with `tenure = 0`,
   i.e. brand new accounts that have not yet been billed).
   `pd.to_numeric(errors="coerce")` turns those blanks into `NaN`.
3. **Median imputation for the 11 gaps.** The median is used rather than the
   mean because `TotalCharges` is right-skewed (median 1397.47 vs mean 2283.30).

   Crucially this happens *inside* the pipeline, so the fill value is learned
   from the training split only and never leaks test information.
4. **`SeniorCitizen` cast to string.** It is stored as `0`/`1` but is a yes/no
   flag, not a quantity, so it is one-hot encoded with the other categoricals
   instead of being scaled as a number.
5. **`Churn` mapped** `Yes` -> `1`, `No` -> `0`.
6. **Numeric features standardised** with `StandardScaler`. This is not
   cosmetic: KNN and SVM are distance and kernel based, so without it the
   `TotalCharges` range (19-8685) would completely dominate `tenure` (0-72).

All preprocessing lives inside a `ColumnTransformer` wrapped in an sklearn
`Pipeline` with each model, so every saved `.joblib` accepts a **raw** CSV
directly and the app never reimplements the training-time transformations.

**Split:** stratified 80/20, `random_state=5331` (taken from my BITS ID 2026AJ05331) - 5,634 train / 1,409 test
rows, both holding the 26.54% churn rate. The test split is saved raw as
`test_data.csv`.

---

## c. GitHub repository

```
https://github.com/dheerajrunthala/telco-churn-ml-classifier
```

---

## d. Model comparison

Scored on the held-out 1,409-row test split. Positive class = churn. Best value
in each column is in bold.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.8105** | **0.8519** | 0.6607 | 0.5882 | **0.6223** | **0.4979** |
| Decision Tree | 0.7388 | 0.6758 | 0.5077 | 0.5321 | 0.5196 | 0.3406 |
| K-Nearest Neighbors | 0.7594 | 0.7799 | 0.5485 | 0.5294 | 0.5388 | 0.3762 |
| Gaussian Naive Bayes | 0.6891 | 0.8212 | 0.4544 | **0.8529** | 0.5929 | 0.4265 |
| SVM (SVC) | 0.8062 | 0.8106 | **0.6624** | 0.5508 | 0.6015 | 0.4784 |

Numbers are read directly from `model/metrics.json`, written by
`train_models.py`. Re-running the script reproduces them exactly.

### Confusion matrices

Test set contains 374 actual churners out of 1,409 customers.

| Model | TN | FP | FN | TP | Customers flagged |
|---|---|---|---|---|---|
| Logistic Regression | 922 | 113 | 154 | 220 | 333 |
| Decision Tree | 842 | 193 | 175 | 199 | 392 |
| K-Nearest Neighbors | 872 | 163 | 176 | 198 | 361 |
| Gaussian Naive Bayes | 652 | 383 | 55 | 319 | 702 |
| SVM (SVC) | 930 | 105 | 168 | 206 | 311 |

---

## e. Observations

| Model | Notes |
|---|---|
| **Logistic Regression** | This was the strongest model overall. It had the best Accuracy, AUC, F1 and MCC scores. It was also quick to train and produced the smallest model file (3 KB). It correctly identified 220 of the 374 customers who churned, while flagging 333 customers in total. |
| **Decision Tree** | This model gave the weakest results overall, with the lowest AUC (0.6758) and MCC (0.3406). I used the default unpruned tree, which appears to make very confident predictions at the leaves. Its accuracy (0.7388) looks acceptable at first, but it is barely above the 73.46% you would get by predicting that nobody churns. Its ability to rank churners is also much weaker than Logistic Regression’s. |
| **K-Nearest Neighbors** | KNN was in the middle across most of the metrics. Its model file was the largest (209 KB) because it keeps all 5,634 training rows. Its performance also depends a lot on the feature scaling used in the pipeline. |
| **Gaussian Naive Bayes** | This model had the lowest accuracy (0.6891), even lower than the 73.46% majority-class baseline. However, it had the highest recall (0.8529) and the second-best AUC (0.8212). It found 319 of the 374 churners, but also flagged 383 customers who did not churn. This trade-off may be useful when missing a churner is more costly than contacting an extra customer. The correlated service features also do not fully satisfy the model's feature-independence assumption. |
| **SVM (SVC)** | SVM was close to Logistic Regression in accuracy (0.8062 compared with 0.8105). It had the best precision (0.6624) and the fewest false positives (105), but it took much longer to train. Since it did not improve the accuracy or overall score, Logistic Regression is the more practical choice here. |
| **Overall winner: Logistic Regression** | I would select Logistic Regression for this dataset. It had the highest MCC (0.4979) and AUC, while also being the simplest and smallest model, and quick to train. MCC is especially useful here because the churn classes are imbalanced. |

**Cross-cutting notes**

- Recall is between 0.52 and 0.59 for every model except Gaussian Naive Bayes.
  Since I used the library defaults and did not add class weighting, most of
  the models miss a noticeable number of churners.
- Accuracy and AUC do not tell exactly the same story. Gaussian Naive Bayes is
  second on AUC but last on accuracy. This shows that it can rank customers
  reasonably well, even though its default predictions produce many false
  positives.
- Neither the RBF SVM nor the non-linear Decision Tree improved on the linear
  Logistic Regression model. For these features, the main churn pattern seems
  to be mostly linear.

---

## f. How to run locally

**Requires Python 3.13** (Streamlit Community Cloud supports up to 3.13).

```bash
# 1. Clone
git clone https://github.com/dheerajrunthala/telco-churn-ml-classifier.git
cd telco-churn-ml-classifier

# 2. Create and activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the models (writes model/ and test_data.csv)
python train_models.py

# 5. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

`model/` is committed, so step 4 is optional - the app runs against the saved
artifacts as-is. Run it if you want to verify the numbers reproduce.

### Using the app

1. Click **Download sample test_data.csv** in the sidebar.
2. Upload that file back through the uploader.
3. Pick a model from the dropdown.
4. **Metrics** shows the six scores; **Confusion Matrix** shows the heatmap and
   classification report; **Compare All 5** scores every model on the same data.

Uploading `test_data.csv` reproduces the table in section (d) exactly. When the
numbers match the held-out results, no difference (delta) is shown on the metric cards.

---

## Live app

```
https://<your-app-name>.streamlit.app
```

---

## Repository structure

```
.
├── app.py                  Streamlit dashboard (loads models, never trains)
├── train_models.py         Training and evaluation script
├── train_models.ipynb      Notebook export of the same script, with outputs
├── requirements.txt        Pinned dependencies
├── test_data.csv           Held-out 20% split, raw columns including Churn
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── gaussian_nb.joblib
│   ├── svm.joblib
│   ├── metrics.json        Test-set scores for all five models
│   └── feature_info.json   Column contract shared by training and the app
└── .streamlit/
    └── config.toml         Indigo light theme
```
