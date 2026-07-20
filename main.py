import pandas as pd

# Load data
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# Dummy prediction: predict the mean price from training data for every row
mean_price = train["price"].mean()

# Build submission dataframe
submission = pd.DataFrame({
    "id": test["id"],
    "price": mean_price
})

# Save submission
submission.to_csv("submission.csv", index=False)

print(f"Submission saved with {len(submission)} rows.")
print(f"Dummy predicted price (mean): {mean_price:.2f}")
