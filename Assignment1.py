import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "price"

# "flight" holds ~870 near-unique codes (e.g. "UK-930"); the airline and
# route columns already capture the useful signal it carries, so it is
# dropped to avoid an explosion of one-hot columns.
CATEGORICAL_COLS = ["airline", "source", "departure", "stops", "arrival", "destination", "class"]
NUMERIC_COLS = ["duration", "days_left"]

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X = train[CATEGORICAL_COLS + NUMERIC_COLS]
y = train[TARGET]
X_test = test[CATEGORICAL_COLS + NUMERIC_COLS]

# Several columns (airline, departure, stops, duration, days_left, ...)
# contain missing values, so both branches impute before the model sees them.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            CATEGORICAL_COLS,
        ),
        ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
    ]
)

model = Pipeline(
    [
        ("preprocess", preprocessor),
        (
            "regressor",
            RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        ),
    ]
)

# Hold out a validation split to sanity-check the r2_score before submitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
val_r2 = r2_score(y_val, model.predict(X_val))
print(f"Validation R2 score: {val_r2:.4f}")

# Refit on the full training set for the final predictions
model.fit(X, y)
test_pred = model.predict(X_test)

# Build submission dataframe
submission = pd.DataFrame({
    "id": test["id"],
    "price": test_pred,
})

# Save submission
submission.to_csv("submission.csv", index=False)

print(f"Submission saved with {len(submission)} rows.")
print(f"Predicted price stats -> min: {test_pred.min():.2f}, "
      f"max: {test_pred.max():.2f}, mean: {test_pred.mean():.2f}")
