import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("final_dataset_multiple_players.csv")

print("Dataset loaded successfully.")
print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# Encode categorical variables if available
# -----------------------------
if "HomeAway" in df.columns:
    df["HomeAwayEncoded"] = df["HomeAway"].map({"Home": 1, "Away": 0})

if "Player" in df.columns:
    df["PlayerEncoded"] = df["Player"].astype("category").cat.codes

# -----------------------------
# Create season-average features by player
# -----------------------------
# These are player-specific season averages
df["SeasonAvgPTS"] = df.groupby("Player")["PTS"].transform("mean")
df["SeasonAvgREB"] = df.groupby("Player")["REB"].transform("mean")
df["SeasonAvgAST"] = df.groupby("Player")["AST"].transform("mean")
df["SeasonAvgTOV"] = df.groupby("Player")["TOV"].transform("mean")
df["SeasonAvgFG_PCT"] = df.groupby("Player")["FG_PCT"].transform("mean")

# -----------------------------
# Create performance-vs-season-average targets
# -----------------------------
df["PTS_vs_SeasonAvg"] = df["PTS"] - df["SeasonAvgPTS"]
df["REB_vs_SeasonAvg"] = df["REB"] - df["SeasonAvgREB"]
df["AST_vs_SeasonAvg"] = df["AST"] - df["SeasonAvgAST"]
df["TOV_vs_SeasonAvg"] = df["TOV"] - df["SeasonAvgTOV"]
df["FG_PCT_vs_SeasonAvg"] = df["FG_PCT"] - df["SeasonAvgFG_PCT"]

# -----------------------------
# Choose target variable
# Example: predict points above/below season average
# -----------------------------
target = "PTS_vs_SeasonAvg"

# -----------------------------
# Baseline feature set
# Player-only features
# -----------------------------
baseline_features = [
    "MIN",
    "REB",
    "AST",
    "FG_PCT",
    "TOV",
    "SeasonAvgPTS",
    "SeasonAvgREB",
    "SeasonAvgAST",
    "SeasonAvgTOV",
    "SeasonAvgFG_PCT"
]

optional_baseline = [
    "PTS_Roll3",
    "REB_Roll3",
    "AST_Roll3",
    "HomeAwayEncoded",
    "PlayerEncoded"
]

for col in optional_baseline:
    if col in df.columns:
        baseline_features.append(col)

# -----------------------------
# Enhanced feature set
# Player + matchup context
# -----------------------------
enhanced_features = baseline_features.copy()

optional_enhanced = [
    "DefensiveRating",
    "PointsAllowed",
    "ReboundsAllowed",
    "OpponentFGPctAllowed",
    "Opponent3PtPctAllowed",
    "Pace",
    "MatchupDifficulty"
]

for col in optional_enhanced:
    if col in df.columns:
        enhanced_features.append(col)

# -----------------------------
# Keep only columns needed
# -----------------------------
needed_columns = [target] + list(set(baseline_features + enhanced_features))
df = df[needed_columns].copy()

# -----------------------------
# Check missing values
# -----------------------------
print("\nMissing values before cleaning:")
print(df.isna().sum())

# -----------------------------
# Fill missing numeric values with median
# -----------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop remaining missing rows if any
df = df.dropna()

print("\nMissing values after cleaning:")
print(df.isna().sum())

# -----------------------------
# Define X and y
# -----------------------------
y = df[target]
X_baseline = df[baseline_features]
X_enhanced = df[enhanced_features]

print("\nBaseline features used:")
print(baseline_features)

print("\nEnhanced features used:")
print(enhanced_features)

# -----------------------------
# Train-test split
# -----------------------------
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

Xe_train, Xe_test, ye_train, ye_test = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42
)

# -----------------------------
# Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# -----------------------------
# Helper evaluation function
# -----------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_label):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    return {
        "Model": model_name,
        "Features Used": feature_label,
        "Target": target,
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4)
    }

# -----------------------------
# Train and evaluate models
# -----------------------------
results = []

for model_name, model in models.items():
    results.append(
        evaluate_model(
            model, Xb_train, Xb_test, yb_train, yb_test,
            model_name, "Player Only"
        )
    )

for model_name, model in models.items():
    results.append(
        evaluate_model(
            model, Xe_train, Xe_test, ye_train, ye_test,
            model_name, "Player + Matchup Difficulty"
        )
    )

# -----------------------------
# Results table
# -----------------------------
results_df = pd.DataFrame(results)

print("\nModel Results:")
print(results_df)

# -----------------------------
# Save results
# -----------------------------
results_df.to_csv("model_results_vs_season_avg.csv", index=False)
print("\nSaved model_results_vs_season_avg.csv successfully")