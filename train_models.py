"""
Telco Customer Churn - Binary Classification
=============================================
M.Tech ML Assignment | BITS

Trains and compares five classifiers on the IBM Telco Customer Churn dataset:
Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and SVM.

Every model is wrapped in an sklearn Pipeline whose first step is a shared
ColumnTransformer. That design choice matters a lot downstream: the persisted
.joblib files accept a RAW csv (text TotalCharges, Yes/No categoricals and all),
so the Streamlit app never has to re-implement preprocessing - it just calls
pipeline.predict(raw_dataframe).

Run:  python train_models.py
"""

# %%
# ---------------------------------------------------------------------------
# Imports and global configuration
# ---------------------------------------------------------------------------
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# A single seed is reused for the split and for every estimator that accepts
# one, so the whole experiment is reproducible end to end.
RANDOM_STATE = 5331
TEST_SIZE = 0.20

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("model")
TEST_CSV_PATH = Path("test_data.csv")

# Column roles. Declared once here and exported to model/feature_info.json in
# Part 2 so the Streamlit app validates uploads against the same source of truth
# instead of hardcoding column names a second time.
TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]  # pure identifier - no predictive signal

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# SeniorCitizen is stored as 0/1 but it is a yes/no flag, not a quantity, so it
# belongs with the categoricals (see clean_raw below, where it is cast to str).
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


# %%
# ---------------------------------------------------------------------------
# Step 1: load the raw dataset and explore it
# ---------------------------------------------------------------------------
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the dataset exactly as shipped - no cleaning yet, on purpose.

    Reading it raw first lets us *see* the known defects (TotalCharges arriving
    as text, blanks hiding as empty strings rather than NaN) instead of silently
    papering over them.
    """
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    """Print the standard first-look diagnostics for the assignment write-up."""
    print("=" * 78)
    print("PHASE 1 - DATA EXPLORATION")
    print("=" * 78)

    print(f"\n[1] Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\n[2] Column dtypes as loaded:")
    dtype_table = pd.DataFrame(
        {"dtype": df.dtypes.astype(str), "n_unique": df.nunique()}
    )
    print(dtype_table.to_string())

    # pandas reports 0 nulls here because the TotalCharges gaps are whitespace
    # strings, not NaN. We therefore count blanks explicitly as well.
    print("\n[3] Missing values:")
    null_counts = df.isna().sum()
    blank_counts = df.apply(
        lambda col: col.astype(str).str.strip().eq("").sum()
    )
    missing_table = pd.DataFrame(
        {"nan_count": null_counts, "blank_string_count": blank_counts}
    )
    # Only show columns that actually have a problem - keeps the output readable.
    flagged = missing_table[(missing_table > 0).any(axis=1)]
    if flagged.empty:
        print("    No NaN and no blank strings in any column.")
    else:
        print(flagged.to_string())
        print(
            "    -> these blank strings are the 11 known TotalCharges gaps;"
            " pd.to_numeric(errors='coerce') converts them to NaN."
        )

    print(f"\n[4] Class balance of target '{TARGET_COLUMN}':")
    counts = df[TARGET_COLUMN].value_counts()
    pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    balance_table = pd.DataFrame({"count": counts, "percent": pcts.round(2)})
    print(balance_table.to_string())
    print(
        "    -> imbalanced (~26.5% churn). Accuracy alone will look flattering,"
        " which is exactly why we also report AUC, Recall, F1 and MCC."
    )


raw_df = load_raw()
explore(raw_df)


# %%
# ---------------------------------------------------------------------------
# Step 2: cleaning
# ---------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Repair the three known defects in the dataset.

    This is the ONLY cleaning applied anywhere, and app.py mirrors it exactly.
    Everything that needs fitted statistics (median imputation, scaling,
    one-hot vocabulary) is deliberately NOT done here - it lives inside the
    pipeline so it is learned from the training split alone and travels with
    the saved model. Keeping those two concerns apart is what prevents leakage.

    1. drop customerID        - identifier, no signal
    2. TotalCharges -> float  - stored as text; blanks become NaN
    3. SeniorCitizen -> str   - 0/1 flag treated as a category
    4. Churn Yes/No -> 1/0    - positive class (1) = customer churned
    """
    out = df.copy()

    # 1. Drop the identifier if it is present. "if present" matters because the
    #    app must tolerate uploads that already had the column stripped out.
    out = out.drop(columns=[c for c in DROP_COLUMNS if c in out.columns])

    # 2. Coerce TotalCharges. errors="coerce" turns the 11 " " entries into NaN
    #    rather than raising, so the pipeline's median imputer can fill them.
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    # 3. SeniorCitizen as string -> OneHotEncoder sees two categories ("0","1")
    #    instead of StandardScaler treating it as a continuous measurement.
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(str)

    # 4. Encode the target. Mapped (not factorized) so that 1 is guaranteed to
    #    mean "Yes"/churn - every metric below uses pos_label=1.
    if TARGET_COLUMN in out.columns:
        out[TARGET_COLUMN] = (
            out[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .astype("int64")
        )

    return out


clean_df = clean_raw(raw_df)

print("\n" + "=" * 78)
print("PHASE 1 - AFTER CLEANING")
print("=" * 78)
print(f"\nShape: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns "
      f"(customerID dropped)")
print(f"TotalCharges dtype: {clean_df['TotalCharges'].dtype}, "
      f"NaN count: {clean_df['TotalCharges'].isna().sum()} "
      f"(to be median-imputed inside each pipeline)")
print(f"SeniorCitizen dtype: {clean_df['SeniorCitizen'].dtype}, "
      f"categories: {sorted(clean_df['SeniorCitizen'].unique())}")
print(f"{TARGET_COLUMN} encoded: "
      f"{clean_df[TARGET_COLUMN].value_counts().to_dict()} (1 = churn)")


# %%
# ---------------------------------------------------------------------------
# Step 3: the shared preprocessor
# ---------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer shared by all five models.

    Using one identical preprocessor for every model is what makes the
    comparison fair - any difference in the metrics is attributable to the
    classifier, not to differing feature treatment.
    """
    # Numeric branch: impute THEN scale.
    #  - median over mean: TotalCharges is right-skewed, so the median is the
    #    more robust fill value for the 11 gaps.
    #  - StandardScaler is not optional here: KNN and SVM are distance/kernel
    #    based, so unscaled tenure (0-72) vs TotalCharges (0-8685) would let one
    #    feature dominate. It also helps LogisticRegression's solver converge.
    numeric_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Categorical branch: impute THEN one-hot.
    #  - most_frequent is the only sensible strategy for unordered strings.
    #  - handle_unknown="ignore" means an unseen category in an uploaded CSV
    #    becomes an all-zero block instead of crashing the app.
    #  - sparse_output=False is the critical flag: GaussianNB cannot accept a
    #    sparse matrix, and a dense output lets all five models share this one
    #    transformer. The widened matrix is only ~45 columns, so dense is cheap.
    categorical_branch = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_branch, NUMERIC_FEATURES),
            ("cat", categorical_branch, CATEGORICAL_FEATURES),
        ],
        # Named columns only; anything else in an uploaded CSV is ignored
        # rather than being passed through and breaking the feature count.
        remainder="drop",
    )


def build_pipelines() -> tuple[dict[str, Pipeline], dict[str, str]]:
    """One Pipeline per model: preprocessor + classifier.

    Returns (pipelines_by_slug, display_names_by_slug).

    Each gets its own freshly built preprocessor instance (same recipe) so the
    five pipelines are fully independent objects that can be pickled separately.

    probability=True on SVC is required because the assignment asks for AUC via
    predict_proba; it makes SVC fit an internal 5-fold Platt calibration, which
    is why SVM is by far the slowest model here.
    """
    model_specs = {
        "logistic_regression": (
            "Logistic Regression",
            # max_iter raised from the default 100: the one-hot widened matrix
            # needs more iterations for lbfgs to converge cleanly.
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "decision_tree": (
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "knn": (
            "K-Nearest Neighbors",
            KNeighborsClassifier(),  # default k=5; deterministic, no seed needed
        ),
        "gaussian_nb": (
            "Gaussian Naive Bayes",
            GaussianNB(),  # deterministic; needs the dense input configured above
        ),
        "svm": (
            "SVM (SVC)",
            SVC(probability=True, random_state=RANDOM_STATE),
        ),
    }

    return {
        slug: Pipeline(
            steps=[("preprocess", build_preprocessor()), ("classifier", estimator)]
        )
        for slug, (_display, estimator) in model_specs.items()
    }, {slug: display for slug, (display, _est) in model_specs.items()}


pipelines, display_names = build_pipelines()

print("\n" + "=" * 78)
print("PHASE 1 - PIPELINES BUILT")
print("=" * 78)
for slug, name in display_names.items():
    print(f"  {slug:22s} -> {name}")

# Sanity check: fit the shared preprocessor once just to report the post-encoding
# width, confirming the 19 raw features expand as expected.
_probe = build_preprocessor().fit(clean_df[ALL_FEATURES])
print(f"\nFeature count: {len(ALL_FEATURES)} raw "
      f"({len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical)"
      f" -> {len(_probe.get_feature_names_out())} after one-hot encoding")


# %%
# ---------------------------------------------------------------------------
# Step 4: stratified 80/20 split + held-out test csv
# ---------------------------------------------------------------------------
# stratify=y preserves the ~26.5% churn rate in both splits. With an imbalanced
# target an unstratified split could shift the ratio enough to make the test
# metrics noisy and not comparable to the training distribution.
X = clean_df[ALL_FEATURES]
y = clean_df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 78)
print("PHASE 1 - TRAIN/TEST SPLIT")
print("=" * 78)
print(f"\nrandom_state = {RANDOM_STATE}, test_size = {TEST_SIZE}, stratified on target")
split_table = pd.DataFrame(
    {
        "rows": [len(X_train), len(X_test)],
        "churn_count": [int(y_train.sum()), int(y_test.sum())],
        "churn_pct": [
            round(y_train.mean() * 100, 2),
            round(y_test.mean() * 100, 2),
        ],
    },
    index=["train", "test"],
)
print(split_table.to_string())

# Export the held-out 20% as RAW rows, sliced out of raw_df by index. Writing
# the raw version (not the cleaned X_test) is deliberate: it keeps customerID,
# leaves TotalCharges as text with its blanks, and keeps Churn as Yes/No - so
# uploading this file to the app genuinely exercises the full cleaning path and
# proves the saved pipelines really do consume unprocessed input.
test_raw_df = raw_df.loc[X_test.index]
test_raw_df.to_csv(TEST_CSV_PATH, index=False)

print(f"\nWrote {TEST_CSV_PATH} -> {test_raw_df.shape[0]} rows x "
      f"{test_raw_df.shape[1]} raw columns")
print(f"  TotalCharges dtype in file: {test_raw_df['TotalCharges'].dtype} (still text)")
print(f"  {TARGET_COLUMN} values in file: "
      f"{sorted(test_raw_df[TARGET_COLUMN].unique())} (still Yes/No)")
print(f"  blank TotalCharges carried into the test file: "
      f"{test_raw_df['TotalCharges'].astype(str).str.strip().eq('').sum()}")

print("\nPhase 1 complete.")
