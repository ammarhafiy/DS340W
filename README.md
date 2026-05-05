# DS340W
Opponent-Aware NBA Player Performance Prediction Using Matchup Difficulty

## Project Overview

This project predicts NBA player performance using machine learning while incorporating opponent strength through a matchup difficulty feature. The main goal is to determine whether adding opponent-context information improves prediction accuracy compared to using player-only statistics.

The project compares two settings:

1. **Single-player experiment**
   - Uses Devin Booker only.
   - Evaluates whether matchup difficulty improves prediction for one individual player.

2. **Multi-player experiment**
   - Uses eleven NBA players:
     - Devin Booker
     - Kevin Durant
     - De'Aaron Fox
     - Paolo Banchero
     - Karl-Anthony Towns
     - Brandon Ingram
     - Tyrese Maxey
     - Jamal Murray
     - Pascal Siakam
     - LaMelo Ball
     - Alperen Sengun

The project compares baseline models that use only player-related features against enhanced models that include opponent defensive context and matchup difficulty.

---

## Research Question

The main research question is:

**Can NBA player performance prediction be improved by incorporating matchup difficulty based on opponent defensive statistics?**

A related goal is to evaluate whether matchup difficulty is more useful in a single-player dataset or a larger multi-player dataset.

---

## Dataset Description

The project uses game-by-game NBA player statistics and opponent team defensive statistics from the 2025–2026 NBA season.

Player-level statistics were collected using the NBA API `playergamelog` endpoint. Opponent defensive statistics were collected using the NBA API `leaguedashteamstats` endpoint.

The final datasets include player performance statistics, opponent defensive statistics, rolling averages, season averages, and matchup difficulty features.

---

## Install Required Python Packages

Before running the code, install the required Python packages. Run this command first:

pip install pandas numpy scikit-learn nba_api

---

## Files in This Repository

### Data Files

| File | Description |
|---|---|
| `final_dataset.csv` | Final single-player dataset for Devin Booker |
| `final_dataset_multiple_players.csv` | Final multi-player dataset using eleven NBA players |
| `model_results_vs_season_avg.csv` | Model results for the multi-player experiment |
| `model_results_vs_season_avg_single_player.csv` | Model results for the single-player experiment |

### Python Files

| File | Description |
|---|---|
| `final_dataset_single_player.py` | Creates the single-player dataset for Devin Booker |
| `final_dataset_multiple_players.py` | Creates the multi-player dataset for eleven NBA players |
| `model_results_vs_season_avg_single_player.py` | Trains and evaluates models for the single-player dataset |
| `model_results_vs_season_avg.py` | Trains and evaluates models for the multi-player dataset |

---

## Features Used

### Player Features

The baseline models use player-related features such as:

- Minutes played (`MIN`)
- Rebounds (`REB`)
- Assists (`AST`)
- Field goal percentage (`FG_PCT`)
- Turnovers (`TOV`)
- Rolling 3-game points average (`PTS_Roll3`)
- Rolling 3-game rebounds average (`REB_Roll3`)
- Rolling 3-game assists average (`AST_Roll3`)
- Season average points (`SeasonAvgPTS`)
- Season average rebounds (`SeasonAvgREB`)
- Season average assists (`SeasonAvgAST`)
- Season average turnovers (`SeasonAvgTOV`)
- Season average field goal percentage (`SeasonAvgFG_PCT`)
- Home/away indicator

### Opponent Features

The enhanced models include opponent-context features such as:

- Defensive rating (`DefensiveRating`)
- Points allowed (`PointsAllowed`)
- Rebounds allowed (`ReboundsAllowed`)
- Opponent field goal percentage allowed (`OpponentFGPctAllowed`)
- Opponent three-point percentage allowed (`Opponent3PtPctAllowed`)
- Pace (`Pace`)
- Matchup difficulty (`MatchupDifficulty`)

---

## Matchup Difficulty

The matchup difficulty score is an engineered feature designed to summarize how difficult an opponent is defensively.

The score is based on several opponent defensive metrics:

- Defensive rating
- Points allowed
- Rebounds allowed
- Opponent field goal percentage allowed
- Opponent three-point percentage allowed
- Pace

The opponent features are normalized using Min-Max scaling. Since lower defensive values usually represent stronger defense, some scaled values are inverted so that higher values represent more difficult matchups.

The matchup difficulty formula uses weighted defensive components:

```text
MatchupDifficulty =
0.30 * DefensiveRatingHard
+ 0.20 * PointsAllowedHard
+ 0.15 * ReboundsAllowedHard
+ 0.15 * OpponentFGPctAllowedHard
+ 0.10 * Opponent3PtPctAllowedHard
+ 0.10 * PaceHard

