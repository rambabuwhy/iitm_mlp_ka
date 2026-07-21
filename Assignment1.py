"""
Assignment 1 - Flight Price Prediction
Covers the Peer Review Rubric in ReadmeAssignment1.md:
  1. Data types            6. Visualizations (>=3)
  2. Descriptive stats      7. Scaling + encoding
  3. Missing values         8. 7 model types
  4. Duplicates             9. Hyperparameter tuning (3 models)
  5. Outliers               10. Model performance comparison
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

pd.set_option("display.width", 120)

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def save_and_show(path):
    """Save the current figure to disk AND force it to render in the notebook.

    plt.show() alone can silently no-op depending on the active matplotlib
    backend (e.g. when a script is run with `!python file.py` instead of
    directly in a notebook cell). Explicitly embedding the saved PNG via
    IPython.display guarantees the image appears in Kaggle/Jupyter output.
    """
    plt.savefig(path, dpi=100)
    plt.show()
    try:
        from IPython.display import Image, display

        display(Image(filename=str(path)))
    except ImportError:
        pass
    plt.close()

TARGET = "price"
STOPS_ORDER = {"zero": 0, "one": 1, "two_or_more": 2}

# =====================================================================
# 0. Load data
# =====================================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# =====================================================================
# 1. Identify data types of different columns
# =====================================================================
print("=" * 70)
print("1. DATA TYPES")
print("=" * 70)
print(train.dtypes)
print(f"\nShape: {train.shape[0]} rows x {train.shape[1]} columns")

numeric_cols_raw = train.select_dtypes(include=np.number).columns.drop("id").tolist()
categorical_cols_raw = train.select_dtypes(include="object").columns.tolist()
print(f"\nNumeric columns:     {numeric_cols_raw}")
print(f"Categorical columns: {categorical_cols_raw}")

# =====================================================================
# 2. Descriptive statistics of numerical columns
# =====================================================================
print("\n" + "=" * 70)
print("2. DESCRIPTIVE STATISTICS (numeric columns)")
print("=" * 70)
desc = train[numeric_cols_raw].agg(["min", "max", "mean", "median", "std"]).T
print(desc)

# =====================================================================
# 3. Identify and handle missing values
# =====================================================================
print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)
missing = train.isna().sum()
missing_pct = (missing / len(train) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
print(missing_report[missing_report["missing_count"] > 0])
print(
    "\nHandling strategy: categorical columns are imputed with the most "
    "frequent value and numeric columns with the median. Imputation is done "
    "inside the modeling pipeline (fit on the training fold only), so no "
    "information leaks from validation/test into training."
)

# =====================================================================
# 4. Identify and handle duplicates
# =====================================================================
print("\n" + "=" * 70)
print("4. DUPLICATES")
print("=" * 70)
dup_count = int(train.duplicated().sum())
print(f"Number of fully duplicated rows: {dup_count}")
if dup_count > 0:
    train = train.drop_duplicates().reset_index(drop=True)
    print(f"Dropped duplicates. New shape: {train.shape}")
else:
    print("No duplicate rows found - nothing to drop.")

# =====================================================================
# 5. Identify and handle outliers
# =====================================================================
print("\n" + "=" * 70)
print("5. OUTLIERS (IQR method)")
print("=" * 70)
for col in ["duration", "days_left", "price"]:
    s = train[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = s[(s < lower) | (s > upper)]
    print(
        f"{col:<12} {len(outliers):>6} outliers outside [{lower:.2f}, {upper:.2f}] "
        f"({len(outliers) / len(s) * 100:.2f}% of non-null rows)"
    )

print(
    "\nDecision: outliers are RETAINED. High 'price' outliers correspond to "
    "genuine Business-class / long-haul fares rather than data-entry errors, "
    "and several of the models compared below (trees, ensembles) are "
    "naturally robust to such extreme values. Dropping them would discard "
    "legitimate signal the model needs to correctly price expensive tickets."
)

# =====================================================================
# 6. Visualizations (>=3) with insights
# =====================================================================
print("\n" + "=" * 70)
print("6. VISUALIZATIONS -> saved to ./plots")
print("=" * 70)

plt.figure(figsize=(7, 5))
sns.histplot(train[TARGET], bins=50, kde=True)
plt.title("Distribution of Ticket Price")
plt.xlabel("Price")
plt.tight_layout()
save_and_show(PLOTS_DIR / "1_price_distribution.png")
print(
    "[1] Price distribution is right-skewed with a long tail of expensive "
    "fares -> motivates using non-linear/tree-based models over plain "
    "linear regression."
)

plt.figure(figsize=(7, 5))
sns.boxplot(data=train, x="class", y=TARGET)
plt.title("Price by Seat Class")
plt.tight_layout()
save_and_show(PLOTS_DIR / "2_price_by_class.png")
print(
    "[2] Business class tickets are priced several times higher than "
    "Economy and show far wider spread -> 'class' is a strong predictor."
)

plt.figure(figsize=(7, 5))
sns.boxplot(data=train, x="stops", y=TARGET, order=["zero", "one", "two_or_more"])
plt.title("Price by Number of Stops")
plt.tight_layout()
save_and_show(PLOTS_DIR / "3_price_by_stops.png")
print(
    "[3] Non-stop flights tend to be cheaper than flights with one or more "
    "stops, though the gap is smaller than the effect of 'class'."
)

plt.figure(figsize=(7, 5))
sample = train.sample(min(5000, len(train)), random_state=42)
sns.scatterplot(data=sample, x="days_left", y=TARGET, alpha=0.3, s=10)
plt.title("Price vs Days Left Before Departure")
plt.tight_layout()
save_and_show(PLOTS_DIR / "4_price_vs_days_left.png")
print(
    "[4] Prices spike sharply for last-minute bookings (~1-2 days before "
    "departure) and flatten out for tickets booked further in advance."
)

plt.figure(figsize=(6, 5))
sns.heatmap(train[["duration", "days_left", "price"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Numeric Features")
plt.tight_layout()
save_and_show(PLOTS_DIR / "5_correlation_heatmap.png")
print(
    "[5] 'days_left' has a mild negative correlation with price and "
    "'duration' a weak positive one -> numeric features alone explain "
    "little of the price variance, so categorical features carry more "
    "signal."
)

# =====================================================================
# 7. Scale numerical features and encode categorical features
# =====================================================================
print("\n" + "=" * 70)
print("7. FEATURE ENCODING & SCALING")
print("=" * 70)
print(
    "Encoding: 'stops' has a natural order (zero < one < two_or_more), so it "
    "is ordinal-encoded to 0/1/2 instead of one-hot encoded. The remaining "
    "categorical columns (airline, source, departure, arrival, destination, "
    "class) have no inherent order, so they are one-hot encoded.\n"
    "Scaling: StandardScaler is applied to numeric features because several "
    "of the models compared below (Linear/Ridge/Lasso regression, KNN) are "
    "sensitive to feature magnitude/scale; scaling is harmless for the "
    "tree-based models, so one shared preprocessing pipeline is used for all."
)

train["stops_ord"] = train["stops"].map(STOPS_ORDER)
test["stops_ord"] = test["stops"].map(STOPS_ORDER)

NUMERIC_COLS = ["duration", "days_left", "stops_ord"]
CATEGORICAL_COLS = ["airline", "source", "departure", "arrival", "destination", "class"]
# "flight" holds ~870 near-unique codes; airline/route/class already capture
# its useful signal, so it is dropped to avoid an explosion of dummy columns.
# A combined "source_destination" route feature was tested too, but it hurt
# KNN (extra one-hot dimensions worsen distance-based models) and gave no
# measurable R2 gain for the tree ensembles, so it was left out.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            NUMERIC_COLS,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            CATEGORICAL_COLS,
        ),
    ]
)

X = train[NUMERIC_COLS + CATEGORICAL_COLS]
y = train[TARGET]
X_test = test[NUMERIC_COLS + CATEGORICAL_COLS]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def make_pipeline(regressor):
    return Pipeline([("preprocess", preprocessor), ("regressor", regressor)])


# =====================================================================
# 8. Model building - 7 different model types
# =====================================================================
print("\n" + "=" * 70)
print("8. MODEL BUILDING - training 7 model types")
print("=" * 70)

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42, max_iter=5000),
    "KNeighbors": KNeighborsRegressor(n_neighbors=7),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}

val_scores = {}
fitted_pipelines = {}
for name, reg in models.items():
    pipe = make_pipeline(reg)
    t0 = time.time()
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    score = r2_score(y_val, val_pred)
    elapsed = time.time() - t0
    val_scores[name] = score
    fitted_pipelines[name] = pipe
    print(f"{name:<22} R2 = {score:.4f}   ({elapsed:.1f}s)")

# =====================================================================
# 9. Hyperparameter tuning on 3 of the models
# =====================================================================
print("\n" + "=" * 70)
print("9. HYPERPARAMETER TUNING (3 models)")
print("=" * 70)

tuning_configs = {
    "Ridge": (
        Ridge(random_state=42),
        {"regressor__alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]},
    ),
    "DecisionTree": (
        DecisionTreeRegressor(random_state=42),
        {
            "regressor__max_depth": [5, 10, 15, 20, None],
            "regressor__min_samples_leaf": [1, 5, 10, 20],
        },
    ),
    "HistGradientBoosting": (
        HistGradientBoostingRegressor(random_state=42),
        {
            "regressor__max_iter": [150, 200, 300, 400],
            "regressor__max_depth": [None, 6, 10, 15],
            "regressor__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
            "regressor__l2_regularization": [0.0, 0.1, 1.0],
            "regressor__max_leaf_nodes": [15, 31, 63, 127],
        },
    ),
}

tuned_scores = {}
tuned_pipelines = {}
for name, (estimator, param_grid) in tuning_configs.items():
    pipe = make_pipeline(estimator)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=8,
        cv=3,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    val_pred = search.best_estimator_.predict(X_val)
    score = r2_score(y_val, val_pred)
    elapsed = time.time() - t0
    label = f"{name} (tuned)"
    tuned_scores[label] = score
    tuned_pipelines[label] = search.best_estimator_
    print(f"{name:<22} best_params={search.best_params_}")
    print(f"{'':<22} R2 = {score:.4f}   ({elapsed:.1f}s)")

# =====================================================================
# 10. Comparison of model performances
# =====================================================================
print("\n" + "=" * 70)
print("10. MODEL PERFORMANCE COMPARISON (validation R2)")
print("=" * 70)

all_scores = {**val_scores, **tuned_scores}
comparison = pd.Series(all_scores).sort_values(ascending=False)
print(comparison.to_string())

plt.figure(figsize=(9, 6))
comparison.sort_values().plot(kind="barh", color="steelblue")
plt.xlabel("Validation R2 score")
plt.title("Model Performance Comparison")
plt.tight_layout()
save_and_show(PLOTS_DIR / "6_model_comparison.png")
print("\nSaved model comparison chart to plots/6_model_comparison.png")

best_name = comparison.index[0]
print(f"\nBest single model: {best_name} (validation R2 = {comparison.iloc[0]:.4f})")

# =====================================================================
# Bonus: blend the top 2 models with a tuned mixing weight to try to
# beat the best single model. Different model families make different
# errors, so a weighted average can reduce variance below either model
# alone - but only if the mixing weight is chosen carefully, since an
# equal-weight blend with a clearly weaker model can drag scores down.
# =====================================================================
print("\n" + "=" * 70)
print("BONUS: BLENDING TOP 2 MODELS (weight search)")
print("=" * 70)

all_pipelines = {**fitted_pipelines, **tuned_pipelines}
name_a, name_b = comparison.index[0], comparison.index[1]
pred_a = all_pipelines[name_a].predict(X_val)
pred_b = all_pipelines[name_b].predict(X_val)

alphas = np.linspace(0, 1, 21)  # weight on the best model (name_a)
blend_r2 = [r2_score(y_val, alpha * pred_a + (1 - alpha) * pred_b) for alpha in alphas]
best_alpha = alphas[int(np.argmax(blend_r2))]
blend_score = max(blend_r2)
print(f"Best blend: {best_alpha:.2f}*{name_a} + {1 - best_alpha:.2f}*{name_b} -> R2 = {blend_score:.4f}")

if blend_score > comparison.iloc[0]:
    print(f"Blend beats the best single model ({comparison.iloc[0]:.4f}) -> using the blend for submission.")
    final_name = f"Blend({best_alpha:.2f}*{name_a}+{1 - best_alpha:.2f}*{name_b})"
    use_blend = True
else:
    print(f"Best single model ({best_name}) still wins -> using it for submission.")
    final_name = best_name
    use_blend = False

# =====================================================================
# Final submission using the best-performing approach, refit on all data
# =====================================================================
if use_blend:
    test_pred_a = all_pipelines[name_a].fit(X, y).predict(X_test)
    test_pred_b = all_pipelines[name_b].fit(X, y).predict(X_test)
    test_pred = best_alpha * test_pred_a + (1 - best_alpha) * test_pred_b
else:
    best_pipeline = all_pipelines[best_name]
    best_pipeline.fit(X, y)
    test_pred = best_pipeline.predict(X_test)

submission = pd.DataFrame({"id": test["id"], "price": test_pred})
submission.to_csv("submission.csv", index=False)

print(f"\nSubmission saved with {len(submission)} rows using {final_name}.")
print(
    f"Predicted price stats -> min: {test_pred.min():.2f}, "
    f"max: {test_pred.max():.2f}, mean: {test_pred.mean():.2f}"
)
