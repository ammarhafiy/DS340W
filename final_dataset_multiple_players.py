import time
import pandas as pd
import numpy as np

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from sklearn.preprocessing import MinMaxScaler

season = "2025-26"

player_names = [
    "Devin Booker",
    "Kevin Durant",
    "De'Aaron Fox",
    "Paolo Banchero",
    "Karl-Anthony Towns",
    "Brandon Ingram",
    "Tyrese Maxey",
    "Jamal Murray",
    "Pascal Siakam",
    "LaMelo Ball",
    "Alperen Sengun"
]

# -----------------------------
# Helper function
# -----------------------------
def find_first_existing(df, possible_cols):
    for col in possible_cols:
        if col in df.columns:
            return col
    return None

# -----------------------------
# Step 1: Get player game logs for multiple players
# -----------------------------
all_player_logs = []

for player_name in player_names:
    try:
        player_matches = players.find_players_by_full_name(player_name)

        if not player_matches:
            print(f"Could not find player: {player_name}")
            continue

        player_id = player_matches[0]["id"]

        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            timeout=60
        )

        player_df = gamelog.get_data_frames()[0]

        if player_df.empty:
            print(f"No data found for: {player_name}")
            continue

        player_df = player_df[
            ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "FG_PCT", "TOV"]
        ].copy()

        player_df["Player"] = player_name
        player_df["Opponent"] = player_df["MATCHUP"].apply(lambda x: x.split()[-1])
        player_df["HomeAway"] = player_df["MATCHUP"].apply(
            lambda x: "Home" if "vs." in x else "Away"
        )

        # Convert date and sort properly
        player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"])
        player_df = player_df.sort_values("GAME_DATE").reset_index(drop=True)

        # Rolling stats per player
        player_df["PTS_Roll3"] = player_df["PTS"].shift(1).rolling(3).mean()
        player_df["REB_Roll3"] = player_df["REB"].shift(1).rolling(3).mean()
        player_df["AST_Roll3"] = player_df["AST"].shift(1).rolling(3).mean()

        all_player_logs.append(player_df)
        print(f"Loaded {player_name}")

        # Delay to reduce NBA API rate-limit issues
        time.sleep(2)

    except Exception as e:
        print(f"Error for {player_name}: {e}")
        time.sleep(5)

if not all_player_logs:
    raise ValueError("No player logs were collected. Check season, players, or API connection.")

# Combine all players
player_df = pd.concat(all_player_logs, ignore_index=True)

print("\nCombined player logs:")
print(player_df.head())

# -----------------------------
# Step 2A: Team advanced stats
# -----------------------------
advanced_stats = leaguedashteamstats.LeagueDashTeamStats(
    season=season,
    measure_type_detailed_defense="Advanced",
    per_mode_detailed="PerGame",
    timeout=60
).get_data_frames()[0]

def_rating_col = find_first_existing(advanced_stats, ["DEF_RATING"])
pace_col = find_first_existing(advanced_stats, ["PACE"])

advanced_keep_cols = ["TEAM_ID", "TEAM_NAME"]
if def_rating_col:
    advanced_keep_cols.append(def_rating_col)
if pace_col:
    advanced_keep_cols.append(pace_col)

advanced_df = advanced_stats[advanced_keep_cols].copy()

# -----------------------------
# Step 2B: Team opponent stats
# -----------------------------
opponent_stats = leaguedashteamstats.LeagueDashTeamStats(
    season=season,
    measure_type_detailed_defense="Opponent",
    per_mode_detailed="PerGame",
    timeout=60
).get_data_frames()[0]

pts_col = find_first_existing(opponent_stats, ["PTS", "OPP_PTS", "PTS_ALLOWED"])
reb_col = find_first_existing(opponent_stats, ["REB", "OPP_REB", "REB_ALLOWED"])
fg_pct_col = find_first_existing(opponent_stats, ["FG_PCT", "OPP_FG_PCT"])
fg3_pct_col = find_first_existing(opponent_stats, ["FG3_PCT", "OPP_FG3_PCT"])

opp_keep_cols = ["TEAM_ID", "TEAM_NAME"]
rename_map = {}

if pts_col:
    opp_keep_cols.append(pts_col)
    rename_map[pts_col] = "PointsAllowed"

if reb_col:
    opp_keep_cols.append(reb_col)
    rename_map[reb_col] = "ReboundsAllowed"

if fg_pct_col:
    opp_keep_cols.append(fg_pct_col)
    rename_map[fg_pct_col] = "OpponentFGPctAllowed"

if fg3_pct_col:
    opp_keep_cols.append(fg3_pct_col)
    rename_map[fg3_pct_col] = "Opponent3PtPctAllowed"

opponent_df = opponent_stats[opp_keep_cols].copy()
opponent_df = opponent_df.rename(columns=rename_map)

# -----------------------------
# Step 2C: Add team abbreviations
# -----------------------------
nba_teams = teams.get_teams()
team_map = {t["full_name"]: t["abbreviation"] for t in nba_teams}

advanced_df["Opponent"] = advanced_df["TEAM_NAME"].map(team_map)
opponent_df["Opponent"] = opponent_df["TEAM_NAME"].map(team_map)

team_df = advanced_df.merge(
    opponent_df.drop(columns=["TEAM_NAME"]),
    on=["TEAM_ID", "Opponent"],
    how="inner"
)

if def_rating_col and def_rating_col in team_df.columns:
    team_df = team_df.rename(columns={def_rating_col: "DefensiveRating"})
if pace_col and pace_col in team_df.columns:
    team_df = team_df.rename(columns={pace_col: "Pace"})

print("\nTeam defensive stats:")
print(team_df.head())

# -----------------------------
# Step 3: Merge player + opponent data
# -----------------------------
merged_df = player_df.merge(team_df, on="Opponent", how="left")

# -----------------------------
# Step 4: Fill missing numeric values
# -----------------------------
numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(
    merged_df[numeric_cols].median()
)

# -----------------------------
# Step 5: Normalize opponent features
# -----------------------------
candidate_features = [
    "DefensiveRating",
    "PointsAllowed",
    "ReboundsAllowed",
    "OpponentFGPctAllowed",
    "Opponent3PtPctAllowed",
    "Pace"
]

available_features = [col for col in candidate_features if col in merged_df.columns]

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(merged_df[available_features])

scaled_df = pd.DataFrame(
    scaled_features,
    columns=[f"{col}Scaled" for col in available_features]
)

merged_df = pd.concat([merged_df.reset_index(drop=True), scaled_df], axis=1)

# -----------------------------
# Step 6: Create stronger matchup difficulty
# -----------------------------
hard_components = {}

if "DefensiveRatingScaled" in merged_df.columns:
    hard_components["DefensiveRatingHard"] = 1 - merged_df["DefensiveRatingScaled"]

if "PointsAllowedScaled" in merged_df.columns:
    hard_components["PointsAllowedHard"] = 1 - merged_df["PointsAllowedScaled"]

if "ReboundsAllowedScaled" in merged_df.columns:
    hard_components["ReboundsAllowedHard"] = 1 - merged_df["ReboundsAllowedScaled"]

if "OpponentFGPctAllowedScaled" in merged_df.columns:
    hard_components["OpponentFGPctAllowedHard"] = 1 - merged_df["OpponentFGPctAllowedScaled"]

if "Opponent3PtPctAllowedScaled" in merged_df.columns:
    hard_components["Opponent3PtPctAllowedHard"] = 1 - merged_df["Opponent3PtPctAllowedScaled"]

if "PaceScaled" in merged_df.columns:
    hard_components["PaceHard"] = 1 - merged_df["PaceScaled"]

for name, values in hard_components.items():
    merged_df[name] = values

base_weights = {
    "DefensiveRatingHard": 0.30,
    "PointsAllowedHard": 0.20,
    "ReboundsAllowedHard": 0.15,
    "OpponentFGPctAllowedHard": 0.15,
    "Opponent3PtPctAllowedHard": 0.10,
    "PaceHard": 0.10
}

available_hard_features = [f for f in base_weights if f in merged_df.columns]
total_weight = sum(base_weights[f] for f in available_hard_features)
final_weights = {f: base_weights[f] / total_weight for f in available_hard_features}

merged_df["MatchupDifficulty"] = 0
for feature, weight in final_weights.items():
    merged_df["MatchupDifficulty"] += weight * merged_df[feature]

# -----------------------------
# Step 6B: Create season-average features
# -----------------------------
merged_df["SeasonAvgPTS"] = merged_df.groupby("Player")["PTS"].transform("mean")
merged_df["SeasonAvgREB"] = merged_df.groupby("Player")["REB"].transform("mean")
merged_df["SeasonAvgAST"] = merged_df.groupby("Player")["AST"].transform("mean")
merged_df["SeasonAvgTOV"] = merged_df.groupby("Player")["TOV"].transform("mean")
merged_df["SeasonAvgFG_PCT"] = merged_df.groupby("Player")["FG_PCT"].transform("mean")

# -----------------------------
# Step 6C: Create performance-vs-season-average columns
# -----------------------------
merged_df["PTS_vs_SeasonAvg"] = merged_df["PTS"] - merged_df["SeasonAvgPTS"]
merged_df["REB_vs_SeasonAvg"] = merged_df["REB"] - merged_df["SeasonAvgREB"]
merged_df["AST_vs_SeasonAvg"] = merged_df["AST"] - merged_df["SeasonAvgAST"]
merged_df["TOV_vs_SeasonAvg"] = merged_df["TOV"] - merged_df["SeasonAvgTOV"]
merged_df["FG_PCT_vs_SeasonAvg"] = merged_df["FG_PCT"] - merged_df["SeasonAvgFG_PCT"]

# -----------------------------
# Step 7: Save final dataset
# -----------------------------
merged_df.to_csv("final_dataset_multiple_players.csv", index=False)

print("\nFinal merged dataset:")
print(merged_df.head())
print("\nSaved final_dataset_multiple_players.csv successfully")