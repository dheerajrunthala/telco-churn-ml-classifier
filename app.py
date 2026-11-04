"""
Telco Customer Churn - Model Comparison Dashboard
==================================================
M.Tech ML Assignment 2 | BITS Pilani
Author: Dheeraj Runthala (2026AJ05331)

Streamlit front-end for the five classifiers trained by train_models.py.

This app only ever loads the saved pipelines - it never trains anything. Because
each .joblib holds a full sklearn Pipeline (ColumnTransformer + classifier), an
uploaded raw csv can go straight into pipeline.predict() after four small
cleaning steps.

Run:  streamlit run app.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
matplotlib.use("Agg")

# INFO-level logging to stdout, which is where Streamlit Community Cloud picks
# up app logs. Only counts, names and error reasons are recorded - never any
# content from an uploaded file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [churn-app] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("churn_app")

MODEL_DIR = Path("model")
METRICS_PATH = MODEL_DIR / "metrics.json"
FEATURE_INFO_PATH = MODEL_DIR / "feature_info.json"
SAMPLE_CSV_PATH = Path("test_data.csv")

FOOTER_TEXT = (
    "Built by Dheeraj Runthala (2026AJ05331) · "
    "M.Tech AIML, BITS Pilani · ML Assignment 2"
)

# Metric keys in a fixed display order, paired with the labels shown on the
# st.metric cards and the table headers.
METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}
# Shown as the caption under each card so the six numbers are self-explaining.
METRIC_HELP = {
    "accuracy": "Share of all rows predicted correctly. Flattered by the 73/27 "
                "class split - a model that never predicts churn still scores 0.735.",
    "auc": "Ranking quality across every threshold, from predict_proba. "
           "Independent of where the 0.5 cutoff sits.",
    "precision": "Of the customers flagged as churners, the share that really churned.",
    "recall": "Of the real churners, the share the model caught.",
    "f1": "Harmonic mean of precision and recall.",
    "mcc": "Correlation across the whole confusion matrix. Only scores high when "
           "both classes are predicted well, so it is the fairest single number here.",
}

# One fixed colour per model, chosen to be colour-blind friendly. The colour is
# tied to the model rather than its rank, so sorting the comparison table does
# not repaint the bars.
MODEL_COLORS = {
    "logistic_regression": "#4a3aa7",  # violet
    "decision_tree": "#1baf7a",        # aqua
    "knn": "#eb6834",                  # orange
    "gaussian_nb": "#2a78d6",          # blue
    "svm": "#e87ba4",                  # magenta
}

# Single-hue indigo ramp, light -> dark, for the confusion-matrix heatmap.
# Verified strictly monotonic in lightness, which is what a sequential
# (magnitude) encoding needs.
INDIGO_RAMP = [
    "#E0E7FF", "#C7D2FE", "#A5B4FC", "#818CF8",
    "#6366F1", "#4F46E5", "#4338CA", "#3730A3",
]
INDIGO_CMAP = LinearSegmentedColormap.from_list("indigo", INDIGO_RAMP)

# Recessive ink colours, matching .streamlit/config.toml.
INK = "#0F172A"
INK_MUTED = "#64748B"
GRID = "#E2E8F0"
SURFACE = "#FFFFFF"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn - Model Comparison",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS for things the config.toml theme cannot reach: the metric cards
# get a visible border so they read as cards, and the active tab gets an indigo
# underline instead of the default red.
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] {
          background: #F8FAFC;
          border: 1px solid #E2E8F0;
          border-left: 4px solid #6366F1;
          border-radius: 8px;
          padding: 14px 16px 10px 16px;
      }
      div[data-testid="stMetricValue"] { font-size: 1.7rem; color: #312E81; }
      div[data-testid="stMetricLabel"] { color: #475569; font-weight: 600; }
      button[data-baseweb="tab"] { font-weight: 600; }
      div[data-baseweb="tab-highlight"] { background-color: #6366F1 !important; }
      /* Pinned to the bottom of the window rather than sitting directly under
         the content. On a short page (the pre-upload state) an in-flow footer
         ended up stranded mid-screen with empty space below it. */
      .app-footer {
          position: fixed;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 99;
          background: #FFFFFF;
          border-top: 1px solid #E2E8F0;
          padding: 0.5rem 1rem;
          color: #64748B;
          font-size: 0.8rem;
          text-align: center;
      }
      /* Keep the last element of the page clear of the pinned footer. */
      .stMainBlockContainer, section.main .block-container {
          padding-bottom: 4rem;
      }
      .hero-sub { color: #64748B; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_footer() -> None:
    """Footer shown at the bottom of the page, under whichever tab is open."""
    st.markdown(f'<div class="app-footer">{FOOTER_TEXT}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Loading the saved artifacts
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    """Load the five pipelines plus both json files, once per server process.

    cache_resource (not cache_data) is the right decorator for unpickled model
    objects: they are shared, not copied per session, and they are not
    serialisable the way cache_data expects.
    """
    if not FEATURE_INFO_PATH.exists() or not METRICS_PATH.exists():
        logger.error("Startup failed: %s or %s is missing",
                     FEATURE_INFO_PATH, METRICS_PATH)
        return None, None, None

    feature_info = json.loads(FEATURE_INFO_PATH.read_text())
    baseline_metrics = json.loads(METRICS_PATH.read_text())

    models = {}
    for slug, meta in feature_info["models"].items():
        path = MODEL_DIR / meta["file"]
        if path.exists():
            models[slug] = joblib.load(path)
        else:
            logger.error("Model file missing: %s", path)

    logger.info("Loaded %d model(s) from %s/: %s",
                len(models), MODEL_DIR, ", ".join(models))
    return models, feature_info, baseline_metrics


# ---------------------------------------------------------------------------
# Cleaning - mirrors clean_raw() in train_models.py exactly
# ---------------------------------------------------------------------------
def clean_uploaded(df: pd.DataFrame, feature_info: dict) -> pd.DataFrame:
    """Apply the same four cleaning steps train_models.py applied before fitting.

    This mirrors clean_raw() in train_models.py step for step. The two copies
    are kept in sync by a check anyone can run: uploading test_data.csv must
    reproduce model/metrics.json exactly. Any divergence here moves those
    numbers, so it cannot pass unnoticed.

    Column names come from feature_info.json rather than being retyped, so the
    app and the training script share one source of truth.
    """
    out = df.copy()

    # 1. Drop the identifier if the upload still has it.
    out = out.drop(
        columns=[c for c in feature_info["drop_columns"] if c in out.columns]
    )

    # 2. TotalCharges arrives as text with blank entries; coerce to numeric so
    #    the blanks become NaN for the pipeline's median imputer to fill.
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    # 3. SeniorCitizen is a 0/1 flag, so it is one-hot encoded as a category
    #    rather than scaled as a quantity.
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(str)

    # 4. Encode the target if it is present. 1 means churned, which is the
    #    positive class every metric below is computed against.
    target = feature_info["target_column"]
    if target in out.columns:
        out[target] = (
            out[target].astype(str).str.strip().map(feature_info["target_mapping"])
        )

    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_upload(df: pd.DataFrame, feature_info: dict) -> list[str]:
    """Return a list of human-readable problems; empty list means good to go."""
    problems = []

    if df.empty:
        problems.append("The file parsed correctly but contains no data rows.")
        return problems

    missing = [c for c in feature_info["required_features"] if c not in df.columns]
    if missing:
        problems.append(
            f"**{len(missing)} required column(s) missing:** `"
            + "`, `".join(missing)
            + "`"
        )

    return problems


def check_target(df: pd.DataFrame, feature_info: dict) -> str | None:
    """Return a warning string if the target is unusable, else None."""
    target = feature_info["target_column"]

    if target not in df.columns:
        return (
            f"No **{target}** column found, so there are no true labels to score "
            "against. Showing predictions only - metrics, the confusion matrix "
            "and the model comparison all need ground truth."
        )

    if df[target].isna().any():
        bad = int(df[target].isna().sum())
        return (
            f"**{target}** has {bad} value(s) that are neither `Yes` nor `No`. "
            "Those rows cannot be scored. Fix or remove them and re-upload."
        )

    if df[target].nunique() < 2:
        only = "churned" if df[target].iloc[0] == 1 else "did not churn"
        return (
            f"Every row in **{target}** is the same class (all {only}). "
            "AUC and MCC are undefined on a single-class target."
        )

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_model(pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Six metrics for one pipeline, with churn (1) as the positive class."""
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]  # column 1 = P(churn)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
    }


@st.cache_data(show_spinner="Scoring all five models...")
def score_all_models(file_signature: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Score every model once and reuse the result across all three tabs.

    file_signature is part of the cache key so re-running on the same upload is
    free, while a new file invalidates the cache. Predictions are cached too, so
    the Confusion Matrix tab does not re-run predict().
    """
    models, _info, _baseline = load_artifacts()

    scored = {}
    for slug, pipeline in models.items():
        scored[slug] = {
            "metrics": score_model(pipeline, X, y),
            "y_pred": pipeline.predict(X),
        }
    return scored


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style_axes(ax) -> None:
    """Recessive grid and axes so the data marks carry the chart."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0)


def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Annotated 2x2 heatmap on the single-hue indigo ramp.

    The counts are printed in every cell, so the colour is a secondary cue and
    the figure stays readable in greyscale or for a colour-blind reader.
    """
    fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=120, facecolor=SURFACE)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=INDIGO_CMAP,
        cbar=False,
        square=True,
        linewidths=2,          # 2px surface gap between cells
        linecolor=SURFACE,
        annot_kws={"fontsize": 15, "fontweight": "bold"},
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )

    ax.set_xlabel("Predicted", fontsize=10, color=INK_MUTED, labelpad=10)
    ax.set_ylabel("Actual", fontsize=10, color=INK_MUTED, labelpad=10)
    ax.set_title(model_name, fontsize=12, color=INK, pad=14, fontweight="bold")
    ax.tick_params(colors=INK, length=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Flip the annotation to white on the darker cells so it stays legible.
    threshold = cm.max() * 0.55
    for text, value in zip(ax.texts, cm.ravel()):
        text.set_color(SURFACE if value > threshold else "#1E1B4B")

    fig.tight_layout()
    return fig


def plot_model_comparison(comparison: pd.DataFrame, slugs: list[str],
                          display_names: dict):
    """Grouped bars: six metric groups on one 0-1 axis, five models as series.

    All six metrics share the same 0-1 range, so a single y-axis is correct and
    no second scale is needed. Values are not printed on the 30 bars - the
    comparison table above the chart carries every number.
    """
    fig, ax = plt.subplots(figsize=(12, 4.0), dpi=120, facecolor=SURFACE)

    n_models = len(slugs)
    group_x = np.arange(len(METRIC_KEYS))
    # Leave a 2px-equivalent gap between adjacent bars within a group.
    bar_width = 0.82 / n_models

    for i, slug in enumerate(slugs):
        offsets = group_x - 0.41 + bar_width * (i + 0.5)
        ax.bar(
            offsets,
            [comparison.loc[slug, key] for key in METRIC_KEYS],
            width=bar_width * 0.92,
            color=MODEL_COLORS[slug],
            label=display_names[slug],
            edgecolor=SURFACE,
            linewidth=1.2,
        )

    ax.set_xticks(group_x)
    ax.set_xticklabels([METRIC_LABELS[k] for k in METRIC_KEYS], fontsize=10, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Score", fontsize=10, color=INK_MUTED)
    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=n_models,
        frameon=False,
        fontsize=9,
        labelcolor=INK,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
models, feature_info, baseline_metrics = load_artifacts()

if not models:
    st.error(
        "#### Saved models not found\n\n"
        f"Expected `{FEATURE_INFO_PATH}`, `{METRICS_PATH}` and the `.joblib` "
        "files inside `model/`.\n\n"
        "Run **`python train_models.py`** once to create them, then reload "
        "this page."
    )
    render_footer()
    st.stop()  # logged by load_artifacts() above

display_names = {s: m["display_name"] for s, m in feature_info["models"].items()}
model_slugs = list(feature_info["models"].keys())

with st.sidebar:
    st.markdown("### 📂 Test data")

    uploaded_file = st.file_uploader(
        "Upload a test csv",
        type=["csv"],
        help="Raw Telco columns. TotalCharges may stay as text and a Churn "
             "column of Yes/No enables the metrics.",
    )

    # Lets the user download the held-out test split without having to run
    # training first.
    if SAMPLE_CSV_PATH.exists():
        st.download_button(
            "⬇ Download sample test_data.csv",
            data=SAMPLE_CSV_PATH.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
            help=f"The held-out {feature_info['n_test_rows']}-row stratified "
                 "20% split, with raw columns.",
        )
    else:
        st.caption("Sample csv not found - run train_models.py to generate it.")

    st.markdown("### 🤖 Model")
    selected_slug = st.selectbox(
        "Choose a model",
        options=model_slugs,
        format_func=lambda s: display_names[s],
        help="Metrics and the confusion matrix follow this choice. The Compare "
             "tab always shows all five.",
    )

    st.divider()

    with st.expander("ℹ️ About", expanded=False):
        dist = feature_info["class_distribution_full"]
        st.markdown(
            f"""
**Dataset - IBM Telco Customer Churn**

{dist['No'] + dist['Yes']:,} customers of a fictional Californian telco, with
{feature_info['n_features_raw']} predictors covering demographics (gender,
senior citizen, partner, dependents), the services each customer subscribes to
(phone, internet, online security, streaming), their contract and payment
method, and three numeric fields - tenure in months, monthly charges and total
charges.

The target is **Churn**: did the customer leave? It is imbalanced -
{dist['Yes']:,} churned vs {dist['No']:,} retained
({dist['churn_rate_pct']}% churn), which is why accuracy alone is not enough.

**Cleaning applied**
"""
        )
        for step in feature_info["cleaning_steps"]:
            st.markdown(f"- {step}")

        st.markdown(
            f"""
Models were fitted on a stratified
{100 - int(feature_info['test_size'] * 100)}/{int(feature_info['test_size'] * 100)}
split ({feature_info['n_train_rows']:,} train / {feature_info['n_test_rows']:,}
test rows, `random_state={feature_info['random_state']}`). Each saved file is a
full sklearn Pipeline, so preprocessing travels with the model.

**The five models**

- **Logistic Regression** - linear baseline; weights each feature's
  contribution to the log-odds of churning. Fast and interpretable.
- **Decision Tree** - recursive yes/no splits. Readable rules, but grown
  unpruned it fits the training split hard.
- **K-Nearest Neighbors** - classifies by the majority vote of the 5 closest
  customers. Needs the feature scaling the pipeline applies.
- **Gaussian Naive Bayes** - probabilistic, assuming features are independent
  given the class. That assumption is violated here, so it over-predicts churn:
  high recall, low precision.
- **SVM (SVC)** - fits a maximum-margin boundary with an RBF kernel;
  `probability=True` adds the calibration needed for AUC.
"""
        )

    st.caption(f"scikit-learn {feature_info['sklearn_version']} · "
               f"seed {feature_info['random_state']}")


# ---------------------------------------------------------------------------
# Main - header
# ---------------------------------------------------------------------------
st.title("📉 Telco Customer Churn: Comparing 5 ML Models")
st.markdown(
    '<p class="hero-sub">Upload a test CSV and see how each model scores on it. '
    "Nothing is trained here. The five models were fitted in train_models.py "
    "and are only loaded from the model/ folder.</p>",
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.info(
        "#### Upload a test csv to begin\n\n"
        "Use **⬇ Download sample test_data.csv** in the sidebar to grab the "
        f"held-out {feature_info['n_test_rows']}-row split, then upload it back "
        "here. Uploading that file reproduces the metrics in "
        "`model/metrics.json` exactly, which is also how the app confirms its "
        "cleaning matches the training script's."
    )
    render_footer()
    st.stop()


# ---------------------------------------------------------------------------
# Main - read and validate the upload
# ---------------------------------------------------------------------------
try:
    raw_bytes = uploaded_file.getvalue()
    raw_upload = pd.read_csv(uploaded_file)
except pd.errors.EmptyDataError:
    logger.warning("Upload rejected: file is empty")
    st.error("#### Could not read that file\n\nIt appears to be empty.")
    render_footer()
    st.stop()
except Exception as exc:  # malformed csv, wrong delimiter, bad encoding
    logger.warning("Upload rejected: could not parse as csv (%s)",
                   type(exc).__name__)
    st.error(
        "#### Could not read that file as csv\n\n"
        f"`{type(exc).__name__}: {exc}`\n\n"
        "Check it is a comma-separated csv with a header row."
    )
    render_footer()
    st.stop()

logger.info("Upload received: %d rows, %d columns",
            len(raw_upload), raw_upload.shape[1])

problems = validate_upload(raw_upload, feature_info)
if problems:
    # Column names are part of the file's schema, not its data, so they are
    # safe to log and are what makes a rejection diagnosable.
    missing = [c for c in feature_info["required_features"]
               if c not in raw_upload.columns]
    logger.warning("Upload rejected: validation failed (%d missing column(s)%s)",
                   len(missing),
                   ": " + ", ".join(missing) if missing else "")
    st.error("#### This file cannot be scored\n\n" + "\n\n".join(problems))
    with st.expander("What the file needs"):
        st.markdown(
            f"**{len(feature_info['required_features'])} feature columns**, plus "
            f"a `{feature_info['target_column']}` column of Yes/No to compute "
            "metrics. `customerID` is optional and dropped if present."
        )
        st.code(", ".join(feature_info["all_raw_columns"]), language="text")
        st.caption("Columns found in your file:")
        st.code(", ".join(map(str, raw_upload.columns)), language="text")
    render_footer()
    st.stop()

logger.info("Model selected: %s", selected_slug)

clean_upload = clean_uploaded(raw_upload, feature_info)
target_column = feature_info["target_column"]
X_upload = clean_upload[feature_info["required_features"]]

st.sidebar.success(f"✅ {len(clean_upload):,} rows loaded")

# ---------------------------------------------------------------------------
# Predictions-only path, when there is no usable ground truth
# ---------------------------------------------------------------------------
target_warning = check_target(clean_upload, feature_info)
if target_warning:
    # Log the reason, not the warning prose, so the cases stay greppable.
    target = feature_info["target_column"]
    if target not in clean_upload.columns:
        reason = "target column absent"
    elif clean_upload[target].isna().any():
        reason = f"{int(clean_upload[target].isna().sum())} unmappable label(s)"
    else:
        reason = "single-class target"
    logger.warning("Scoring limited to predictions only: %s", reason)
    st.warning("#### " + target_warning)

    pipeline = models[selected_slug]
    predictions = raw_upload.copy()
    predictions["predicted_churn"] = np.where(
        pipeline.predict(X_upload) == 1, "Yes", "No"
    )
    predictions["churn_probability"] = pipeline.predict_proba(X_upload)[:, 1].round(4)

    flagged = int((predictions["predicted_churn"] == "Yes").sum())
    left, right = st.columns(2)
    left.metric("Rows scored", f"{len(predictions):,}")
    right.metric(
        "Predicted to churn",
        f"{flagged:,}",
        f"{flagged / len(predictions) * 100:.1f}% of uploaded rows",
        delta_color="off",
    )

    st.markdown(f"#### Predictions - {display_names[selected_slug]}")
    st.dataframe(predictions, width="stretch", height=420)
    st.download_button(
        "⬇ Download predictions csv",
        data=predictions.to_csv(index=False).encode(),
        file_name=f"predictions_{selected_slug}.csv",
        mime="text/csv",
    )

    render_footer()
    st.stop()


# ---------------------------------------------------------------------------
# Full scoring path
# ---------------------------------------------------------------------------
y_upload = clean_upload[target_column].astype("int64")

file_signature = f"{uploaded_file.name}:{len(raw_bytes)}:{hash(raw_bytes)}"
scored = score_all_models(file_signature, X_upload, y_upload)
logger.info("Scored %d model(s) on %d rows", len(scored), len(y_upload))

comparison = pd.DataFrame(
    {slug: scored[slug]["metrics"] for slug in model_slugs}
).T.loc[model_slugs, METRIC_KEYS]

tab_metrics, tab_confusion, tab_compare = st.tabs(
    ["Metrics", "Confusion Matrix", "Compare All 5"]
)


# --- Tab 1: metrics ---------------------------------------------------------
with tab_metrics:
    st.markdown(f"### {display_names[selected_slug]}")
    st.caption(
        f"Scored on {len(y_upload):,} uploaded rows · "
        f"{int(y_upload.sum()):,} actual churners "
        f"({y_upload.mean() * 100:.1f}%) · positive class = churn"
    )

    selected_metrics = scored[selected_slug]["metrics"]
    reference = baseline_metrics[selected_slug]

    # Two rows of three cards. The delta compares the uploaded data against the
    # held-out test score in metrics.json, so uploading test_data.csv shows a
    # delta of zero everywhere - a quick confirmation nothing has drifted.
    for row_keys in (METRIC_KEYS[:3], METRIC_KEYS[3:]):
        for column, key in zip(st.columns(3), row_keys):
            delta = selected_metrics[key] - reference[key]
            # Only show a delta when the score actually differs. Passing a
            # string for the "matches" case made st.metric draw an up-arrow
            # next to it, which read as an improvement rather than equality.
            column.metric(
                METRIC_LABELS[key],
                f"{selected_metrics[key]:.4f}",
                delta=(f"{delta:+.4f} vs held-out test"
                       if abs(delta) > 5e-5 else None),
                delta_color="off",
                help=METRIC_HELP[key],
            )

    st.divider()
    st.caption(
        "Deltas compare your upload against this model's score on the original "
        "held-out split recorded in `model/metrics.json`."
    )


# --- Tab 2: confusion matrix + classification report -----------------------
with tab_confusion:
    st.markdown(f"### {display_names[selected_slug]}")

    y_pred_selected = scored[selected_slug]["y_pred"]
    cm = confusion_matrix(y_upload, y_pred_selected, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        cm_figure = plot_confusion_matrix(cm, display_names[selected_slug])
        st.pyplot(cm_figure, width="stretch")
        plt.close(cm_figure)  # Streamlit reruns on every interaction; without
                              # this, figures accumulate until matplotlib warns.

    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_upload,
            y_pred_selected,
            labels=[0, 1],
            target_names=["No churn (0)", "Churn (1)"],
            output_dict=True,
            zero_division=0,
        )
        # "accuracy" comes back as a bare float rather than a dict of the four
        # columns. Left in, pandas broadcasts it across precision/recall/f1/
        # support and the support column stops being a count, so pull it out and
        # report it on its own.
        overall_accuracy = report.pop("accuracy", None)
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format(
                {"precision": "{:.4f}", "recall": "{:.4f}",
                 "f1-score": "{:.4f}", "support": "{:,}"}
            ),
            width="stretch",
        )
        if overall_accuracy is not None:
            st.caption(f"Overall accuracy: **{overall_accuracy:.4f}** "
                       f"over {len(y_upload):,} rows")

        st.markdown("#### What the four cells mean")
        st.markdown(
            f"""
| Cell | Count | Reading |
|---|---|---|
| True negatives | **{tn:,}** | stayed, predicted to stay |
| False positives | **{fp:,}** | stayed, but flagged as churn risk |
| False negatives | **{fn:,}** | churned and were missed |
| True positives | **{tp:,}** | churned and were caught |

This model flags **{tp + fp:,}** of {len(y_upload):,} customers as churn risks
and catches **{tp:,}** of the **{tp + fn:,}** real churners.
"""
        )


# --- Tab 3: compare all five models ----------------------------------------
with tab_compare:
    st.markdown("### Every model, same data")
    st.caption(
        f"All five pipelines scored on the same {len(y_upload):,} rows with the "
        "same preprocessing. The best value in each column is in bold."
    )

    table = comparison.rename(index=display_names)
    table.columns = [METRIC_LABELS[k] for k in METRIC_KEYS]
    table.index.name = "ML Model Name"

    # Sorted by MCC because it is the metric least distorted by the class
    # imbalance. Bar colours stay keyed to the model, so sorting here does not
    # change which colour a model wears in the chart below.
    table = table.sort_values("MCC", ascending=False)

    st.dataframe(
        table.style.format("{:.4f}").highlight_max(axis=0, props="font-weight:700;"),
        width="stretch",
    )

    comparison_figure = plot_model_comparison(comparison, model_slugs, display_names)
    st.pyplot(comparison_figure, width="stretch")
    plt.close(comparison_figure)

    best_slug = comparison["mcc"].idxmax()
    st.success(
        f"**Best MCC: {display_names[best_slug]}** "
        f"({comparison.loc[best_slug, 'mcc']:.4f}). MCC is the fairest single "
        "metric here because it only scores high when both the churners and the "
        "non-churners are predicted well."
    )

    with st.expander("Per-metric winners"):
        winners = pd.DataFrame(
            {
                "Metric": [METRIC_LABELS[k] for k in METRIC_KEYS],
                "Best model": [display_names[comparison[k].idxmax()]
                               for k in METRIC_KEYS],
                "Score": [f"{comparison[k].max():.4f}" for k in METRIC_KEYS],
            }
        )
        st.dataframe(winners, width="stretch", hide_index=True)


# One footer for the whole page - it is position:fixed, so it shows under
# whichever tab is open.
render_footer()
