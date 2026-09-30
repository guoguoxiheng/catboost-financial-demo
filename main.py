import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Generate simulated structured financial data
# ============================================================

RANDOM_SEED = 42
N_SAMPLES = 5000

np.random.seed(RANDOM_SEED)

df = pd.DataFrame({
    "rsi": np.random.uniform(20, 80, N_SAMPLES),
    "volume": np.random.uniform(50, 200, N_SAMPLES),
    "macd": np.random.uniform(-2, 2, N_SAMPLES),
    "volatility": np.random.uniform(0.5, 5, N_SAMPLES),
})

# Hidden rule used only to generate the simulated target.
# CatBoost does not receive this formula during training.
score = (
    0.04 * (df["rsi"] - 50)
    + 0.015 * (df["volume"] - 100)
    + 0.8 * df["macd"]
    - 0.2 * df["volatility"]
    + np.random.normal(0, 1.5, N_SAMPLES)
)

df["target"] = (score > 0).astype(int)


# ============================================================
# 2. Prepare features and target
# ============================================================

FEATURES = [
    "rsi",
    "volume",
    "macd",
    "volatility",
]

X = df[FEATURES]
y = df["target"]


# ============================================================
# 3. Train / Validation / Test split
# ============================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=RANDOM_SEED,
)

print("=" * 50)
print("Dataset")
print("=" * 50)
print(f"Total samples:      {len(df)}")
print(f"Train samples:      {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")


# ============================================================
# 4. Train CatBoost
# ============================================================

model = CatBoostClassifier(
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=RANDOM_SEED,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    use_best_model=True,
)


# ============================================================
# 5. Model evaluation
# ============================================================

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, val_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\n" + "=" * 50)
print("Model Performance")
print("=" * 50)

print(f"Best iteration:      {model.get_best_iteration()}")
print(f"Train Accuracy:      {train_accuracy:.2%}")
print(f"Validation Accuracy: {val_accuracy:.2%}")
print(f"Test Accuracy:       {test_accuracy:.2%}")


# ============================================================
# 6. Feature importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.get_feature_importance(),
}).sort_values(
    "importance",
    ascending=False,
)

print("\n" + "=" * 50)
print("Feature Importance")
print("=" * 50)

print(importance_df.to_string(index=False))


# ============================================================
# 7. Save experiment results
# ============================================================

metrics_df = pd.DataFrame({
    "dataset": [
        "train",
        "validation",
        "test",
    ],
    "accuracy": [
        train_accuracy,
        val_accuracy,
        test_accuracy,
    ],
})

metrics_df.to_csv(
    "results/model_metrics.csv",
    index=False,
)

importance_df.to_csv(
    "results/feature_importance.csv",
    index=False,
)


# ============================================================
# 8. Save feature importance chart
# ============================================================

plot_df = importance_df.sort_values(
    "importance",
    ascending=True,
)

plt.figure(figsize=(8, 5))

plt.barh(
    plot_df["feature"],
    plot_df["importance"],
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=200,
)

plt.close()


print("\nResults saved to results/")