import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker
import joblib


# ----------------------------
# Name normalization + fallback keys
# ----------------------------
_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", flags=re.IGNORECASE)

def norm_name(s: str) -> str:
    """Strong normalization to maximize exact matches across data sources."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()

    # remove suffixes (jr, sr, ii, iii...)
    s = _SUFFIX_RE.sub("", s)

    # normalize punctuation
    s = s.replace("’", "").replace("'", "").replace(".", "")
    s = s.replace("-", " ")

    # remove non-letters/spaces
    s = re.sub(r"[^a-z\s]", "", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def last_first_initial_key(full_name: str) -> str:
    """
    Fallback key: last name + first initial (helps when full names don't match exactly).
    Example: "carmelo anthony" -> "anthony_c"
    """
    n = norm_name(full_name)
    if not n:
        return ""
    parts = n.split()
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last = parts[-1]
    return f"{last}_{first[0]}" if first else last


def pick_first_existing(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ----------------------------
# Ranking metrics
# ----------------------------
def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    rel = y_true[order][:k]
    if rel.size == 0:
        return 0.0

    gains = (2 ** rel - 1.0)
    discounts = np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum(gains / discounts))

    ideal = np.sort(y_true)[::-1][:k]
    ideal_gains = (2 ** ideal - 1.0)
    idcg = float(np.sum(ideal_gains / discounts)) if ideal.size else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def mean_group_ndcg(model, df: pd.DataFrame, feature_cols, group_col: str, label_col: str, k: int) -> float:
    scores = []
    for _, g in df.groupby(group_col):
        Xg = g[feature_cols].to_numpy()
        yg = g[label_col].to_numpy()
        sg = model.predict(Xg)
        scores.append(ndcg_at_k(yg, sg, k=k))
    return float(np.mean(scores)) if scores else 0.0


def group_sizes(df: pd.DataFrame, group_col: str) -> np.ndarray:
    # must be in same order as rows passed to fit()
    return df.groupby(group_col).size().to_numpy()


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--college_csv", required=True)
    parser.add_argument("--draft_xlsx", required=True)
    parser.add_argument("--outdir", default="outputs_parent")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ----------------------------
    # Load College CSV
    # ----------------------------
    college = pd.read_csv(args.college_csv, low_memory=False)

    # Your file has: player_name, yr, ...
    name_col = pick_first_existing(college, ["player_name", "Player", "player", "PLAYER", "Name", "name"])
    yr_col   = pick_first_existing(college, ["yr", "Yr", "YR", "year", "Year", "Season", "season"])

    if name_col is None:
        raise ValueError(f"Could not find player name column in college CSV. Columns: {list(college.columns)[:40]}")
    if yr_col is None:
        # not strictly needed for ranking (we group by draft year), so we won't hard fail
        print("WARNING: Could not find 'yr' column in college CSV. Continuing without it.")
        yr_col = None

    college = college.copy()
    college["player_norm"] = college[name_col].astype(str).apply(norm_name)
    college["lf_key"] = college[name_col].astype(str).apply(last_first_initial_key)

    # ----------------------------
    # Load Draft XLSX
    # ----------------------------
    drafted = pd.read_excel(args.draft_xlsx)

    # Your file columns: PLAYER, YEAR, OVERALL
    d_name = pick_first_existing(drafted, ["PLAYER", "Player", "player", "Name", "name"])
    d_year = pick_first_existing(drafted, ["YEAR", "Year", "year", "DraftYear", "draft_year"])
    d_pick = pick_first_existing(drafted, ["OVERALL", "Overall", "overall", "DraftPick", "draft_pick", "Pick", "pick"])

    if d_name is None or d_year is None or d_pick is None:
        raise ValueError(
            "Could not auto-detect columns in drafted XLSX.\n"
            "Need: (player name, draft year, draft pick).\n"
            f"Columns (first 40): {list(drafted.columns)[:40]}"
        )

    drafted = drafted.copy()
    drafted["player_norm"] = drafted[d_name].astype(str).apply(norm_name)
    drafted["lf_key"] = drafted[d_name].astype(str).apply(last_first_initial_key)
    drafted["draft_year"] = pd.to_numeric(drafted[d_year], errors="coerce")
    drafted["draft_pick"] = pd.to_numeric(drafted[d_pick], errors="coerce")
    drafted = drafted.dropna(subset=["player_norm", "draft_year", "draft_pick"]).copy()
    drafted["draft_year"] = drafted["draft_year"].astype(int)
    drafted["draft_pick"] = drafted["draft_pick"].astype(int)

    # ----------------------------
    # Diagnostics before merge
    # ----------------------------
    print("=====================================")
    print("Diagnostics")
    print("College rows:", len(college), "| unique players:", college["player_norm"].nunique())
    print("Draft rows:", len(drafted), "| unique players:", drafted["player_norm"].nunique())
    common_exact = set(college["player_norm"].unique()) & set(drafted["player_norm"].unique())
    print("Common exact-name matches:", len(common_exact))
    print("=====================================")

    # ----------------------------
    # Merge strategy:
    # 1) Exact name match
    # 2) If too small, fallback to last+first-initial key
    # ----------------------------
    df = college.merge(
        drafted[["player_norm", "draft_year", "draft_pick"]],
        on="player_norm",
        how="inner"
    )

    # If merge is suspiciously small, try fallback key merge
    if len(df) < 500:
        print(f"Exact merge produced only {len(df)} rows. Trying fallback merge (last name + first initial)...")
        df2 = college.merge(
            drafted[["lf_key", "draft_year", "draft_pick"]],
            on="lf_key",
            how="inner"
        )

        # Prefer fallback only if it improves a lot
        if len(df2) > len(df) * 2:
            df = df2.copy()
            print(f"Fallback merge successful: {len(df)} rows.")
        else:
            print(f"Fallback merge not much better ({len(df2)} rows). Keeping exact merge ({len(df)} rows).")

    if len(df) == 0:
        raise ValueError(
            "Merge produced 0 rows (no matching players). "
            "This is usually a naming mismatch between datasets."
        )

    # Keep 1 row per (player, draft_year).
    # Your college 'yr' is year-in-school, not season year, so we do NOT compare to draft_year.
    if yr_col is not None:
        # If yr is numeric-ish, take max; if it's strings (Fr/So/Jr/Sr), just keep last occurrence
        try:
            tmp = pd.to_numeric(df[yr_col], errors="coerce")
            df["_yr_num"] = tmp
            df = df.sort_values(["draft_year", "player_norm" if "player_norm" in df.columns else "lf_key", "_yr_num"])
            df = df.groupby(["draft_year", "player_norm" if "player_norm" in df.columns else "lf_key"], as_index=False).tail(1)
            df = df.drop(columns=["_yr_num"])
        except Exception:
            df = df.sort_values(["draft_year"])
            df = df.groupby(["draft_year", "player_norm" if "player_norm" in df.columns else "lf_key"], as_index=False).tail(1)
    else:
        df = df.sort_values(["draft_year"])
        df = df.groupby(["draft_year", "player_norm" if "player_norm" in df.columns else "lf_key"], as_index=False).tail(1)

    # ----------------------------
    # Build relevance labels for Learning-to-Rank
    # Higher relevance = better (lower draft_pick)
    # Make integer labels 0..31 (safe for ndcg)
    # ----------------------------
    df["max_pick"] = df.groupby("draft_year")["draft_pick"].transform("max")
    denom = (df["max_pick"] - 1).replace(0, 1)
    df["relevance"] = (31 - np.floor(31 * (df["draft_pick"] - 1) / denom)).astype(int)

    # ----------------------------
    # Feature columns: numeric only
    # ----------------------------
    exclude = {"draft_year", "draft_pick", "relevance", "max_pick"}
    if "player_norm" in df.columns:
        exclude.add("player_norm")
    if "lf_key" in df.columns:
        exclude.add("lf_key")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    # ----------------------------
    # Train/Test/Validation split by draft_year (70/20/10)
    # ----------------------------
    years = df["draft_year"].unique()
    if len(years) < 3:
        raise ValueError(f"Not enough draft_year groups ({len(years)}) to create train/test/val splits.")

    train_years, temp_years = train_test_split(years, test_size=0.30, random_state=42)
    test_years, val_years = train_test_split(temp_years, test_size=1/3, random_state=42)  # 20% / 10%

    train_df = df[df["draft_year"].isin(train_years)].sort_values("draft_year").reset_index(drop=True)
    test_df  = df[df["draft_year"].isin(test_years)].sort_values("draft_year").reset_index(drop=True)
    val_df   = df[df["draft_year"].isin(val_years)].sort_values("draft_year").reset_index(drop=True)

    # ----------------------------
    # Impute missing feature values using TRAIN medians
    # ----------------------------
    medians = {}
    for c in feature_cols:
        med = float(train_df[c].median()) if train_df[c].notna().any() else 0.0
        medians[c] = med
        train_df[c] = train_df[c].fillna(med)
        test_df[c]  = test_df[c].fillna(med)
        val_df[c]   = val_df[c].fillna(med)

    # ----------------------------
    # Train XGBoost Ranker
    # ----------------------------
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["relevance"].to_numpy()
    g_train = group_sizes(train_df, "draft_year")

    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["relevance"].to_numpy()
    g_test = group_sizes(test_df, "draft_year")

    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg",
        ndcg_exp_gain=False,
        n_estimators=350,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        tree_method="hist",
        n_jobs=4
    )

    model.fit(
        X_train, y_train,
        group=g_train,
        eval_set=[(X_test, y_test)],
        eval_group=[g_test],
        verbose=False
    )

    # ----------------------------
    # Evaluate
    # ----------------------------
    test_ndcg = mean_group_ndcg(model, test_df, feature_cols, "draft_year", "relevance", args.k)
    val_ndcg  = mean_group_ndcg(model, val_df, feature_cols, "draft_year", "relevance", args.k)

    print("=====================================")
    print("Parent-Paper Style Draft Ranking (Local Data)")
    print(f"College file: {args.college_csv}")
    print(f"Draft file:   {args.draft_xlsx}")
    print(f"Draft years -> Train={len(train_years)} Test={len(test_years)} Val(unseen)={len(val_years)}")
    print(f"Rows -> Train={len(train_df)} Test={len(test_df)} Val={len(val_df)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Test mean NDCG@{args.k}: {test_ndcg:.4f}")
    print(f"Val  mean NDCG@{args.k}: {val_ndcg:.4f}")
    print("=====================================")

    # ----------------------------
    # Save outputs
    # ----------------------------
    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(model, os.path.join(args.outdir, "xgb_ranker.joblib"))
    pd.Series(feature_cols).to_csv(os.path.join(args.outdir, "feature_cols.csv"), index=False)
    pd.Series(medians).to_csv(os.path.join(args.outdir, "train_medians.csv"))

    train_df.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.outdir, "test.csv"), index=False)
    val_df.to_csv(os.path.join(args.outdir, "validation_unseen.csv"), index=False)

    print(f"Saved model + splits to: {args.outdir}")


if __name__ == "__main__":
    main()
