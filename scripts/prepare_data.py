import pandas as pd
import numpy as np
import os
import requests
import re
import time

start_total = time.time()

# -----------------------------
# Helper functions
# -----------------------------
def download_file_if_missing(url, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {url} (status code {response.status_code})")

def normalize_name(name):
    if pd.isna(name): return name
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name

def parse_match_id(match_id):
    match_id = str(match_id)
    m = re.search(r'-(MS|WS)(\d+)$', match_id)
    if m:
        best_of = 5 if m.group(1) == 'MS' else 3
        round_num = int(m.group(2)[0])
        return pd.Series([best_of, round_num])
    m2 = re.search(r'(\d{4})$', match_id)
    if m2:
        last4 = m2.group(1)
        gender_digit = int(last4[0])
        best_of = 5 if gender_digit == 1 else 3
        round_num = int(last4[1])
        return pd.Series([best_of, round_num])
    return pd.Series([np.nan, np.nan])

# -----------------------------
# Config
# -----------------------------
DEBUG = False
if DEBUG:
    years = [2024]
    slams = ['wimbledon']
else:
    years = list(range(2020, 2025))
    slams = ['ausopen', 'frenchopen', 'wimbledon', 'usopen']

points_df_list = []
matches_df_list = []

# -----------------------------
# Load data
# -----------------------------
start = time.time()
for year in years:
    for slam in slams:
        points_file = f"{year}-{slam}-points.csv"
        matches_file = f"{year}-{slam}-matches.csv"
        points_path = f"data/raw/{points_file}"
        matches_path = f"data/raw/{matches_file}"
        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"

        try:
            download_file_if_missing(base_url + points_file, points_path)
            download_file_if_missing(base_url + matches_file, matches_path)

            print(f"Loading {year} {slam}...")
            points_df = pd.read_csv(points_path)
            matches_df = pd.read_csv(matches_path)

            points_df["tournament"] = slam
            points_df["year"] = year
            matches_df["tournament"] = slam
            matches_df["year"] = year

            points_df_list.append(points_df)
            matches_df_list.append(matches_df)

        except Exception as e:
            print(f"Failed to load {year} {slam}: {e}")

pbp_df = pd.concat(points_df_list, ignore_index=True)
match_df = pd.concat(matches_df_list, ignore_index=True)
print(f"Data loading took {time.time() - start:.2f} seconds")

# -----------------------------
# Merge
# -----------------------------
merged = pd.merge(
    pbp_df,
    match_df[['match_id', 'player1', 'player2', 'event_name']],
    on='match_id',
    how='left'
)

mask = merged["PointNumber"].astype(str).str.match(r"^\d+$")
print("Filtered out rows with non-numeric PointNumber:", (~mask).sum())
merged = merged[mask]

merged["SetNo"] = merged["SetNo"].astype(int)
merged["GameNo"] = merged["GameNo"].astype(int)
merged["PointNumber"] = merged["PointNumber"].astype(int)

merged = merged.sort_values(
    by=["match_id", "SetNo", "GameNo", "PointNumber"]
).reset_index(drop=True)

# -----------------------------
# Vectorized Score Reconstruction
# -----------------------------
# 1. Point indicators
merged['p1_point'] = (merged['PointWinner'] == 1).astype(int)
merged['p2_point'] = (merged['PointWinner'] == 2).astype(int)

# -----------------------------
# 2. Cumulative points per game
merged['p1_game_points'] = merged.groupby(['match_id','SetNo','GameNo'])['p1_point'].cumsum()
merged['p2_game_points'] = merged.groupby(['match_id','SetNo','GameNo'])['p2_point'].cumsum()

# -----------------------------
# 3. Pre-point cumulative points
merged['p1_game_points_pre'] = merged.groupby(['match_id','SetNo','GameNo'])['p1_game_points'].shift(1).fillna(0).astype(int)
merged['p2_game_points_pre'] = merged.groupby(['match_id','SetNo','GameNo'])['p2_game_points'].shift(1).fillna(0).astype(int)

# -----------------------------
# 4. Determine game winners (last point of the game)
last_point_mask = merged.groupby(['match_id','SetNo','GameNo'])['PointNumber'].transform('idxmax')
merged['p1_game_won'] = 0
merged['p2_game_won'] = 0

merged.loc[last_point_mask, 'p1_game_won'] = ((merged.loc[last_point_mask,'p1_game_points'] >= 4) & 
                                              (merged.loc[last_point_mask,'p1_game_points'] - merged.loc[last_point_mask,'p2_game_points'] >= 2)).astype(int)
merged.loc[last_point_mask, 'p2_game_won'] = ((merged.loc[last_point_mask,'p2_game_points'] >= 4) & 
                                              (merged.loc[last_point_mask,'p2_game_points'] - merged.loc[last_point_mask,'p1_game_points'] >= 2)).astype(int)
# -----------------------------
# 5b. Reset in-set cumulative games
# -----------------------------
# Cumulative games per set, pre-point, properly reset
def cumulative_games_pre(group):
    return group['p1_game_won'].cumsum().shift(fill_value=0)

merged['P1_Games_Won'] = merged.groupby(['match_id','SetNo'], group_keys=False).apply(
    lambda g: g['p1_game_won'].cumsum().shift(fill_value=0).astype(int)
)
merged['P2_Games_Won'] = merged.groupby(['match_id','SetNo'], group_keys=False).apply(
    lambda g: g['p2_game_won'].cumsum().shift(fill_value=0).astype(int)
)

# Cumulative games including current game
for s in range(1,6):
    mask = merged['SetNo'] == s
    merged.loc[mask, f'set{s}_p1'] = merged.loc[mask, 'P1_Games_Won'] + merged.loc[mask, 'p1_game_won']
    merged.loc[mask, f'set{s}_p2'] = merged.loc[mask, 'P2_Games_Won'] + merged.loc[mask, 'p2_game_won']


# 5b. Cumulative sets won per match (pre-point)
last_point_set_mask = merged.groupby(['match_id','SetNo'])['PointNumber'].transform('idxmax')

merged['P1_Sets_Won'] = 0
merged['P2_Sets_Won'] = 0

merged.loc[last_point_set_mask, 'P1_Sets_Won'] = np.where(
    merged.loc[last_point_set_mask, 'P1_Games_Won'] + merged.loc[last_point_set_mask, 'p1_game_won'] > 
    merged.loc[last_point_set_mask, 'P2_Games_Won'] + merged.loc[last_point_set_mask, 'p2_game_won'], 1, 0
)
merged.loc[last_point_set_mask, 'P2_Sets_Won'] = np.where(
    merged.loc[last_point_set_mask, 'P2_Games_Won'] + merged.loc[last_point_set_mask, 'p2_game_won'] > 
    merged.loc[last_point_set_mask, 'P1_Games_Won'] + merged.loc[last_point_set_mask, 'p1_game_won'], 1, 0
)

merged['P1_Sets_Won'] = merged.groupby('match_id')['P1_Sets_Won'].transform(lambda x: x.shift(fill_value=0).cumsum().astype(int))
merged['P2_Sets_Won'] = merged.groupby('match_id')['P2_Sets_Won'].transform(lambda x: x.shift(fill_value=0).cumsum().astype(int))

# -----------------------------
# -----------------------------
# 6. Identify tiebreaks per point
def mark_tiebreak(group):
    group = group.copy()
    # Start tiebreak if cumulative games = 6–6 in the set
    group['is_tiebreak'] = False
    tiebreak_start = ((group['P1_Games_Won'] == 6) & (group['P2_Games_Won'] == 6))
    group.loc[tiebreak_start, 'is_tiebreak'] = True
    return group

merged = merged.groupby(['match_id','SetNo']).apply(mark_tiebreak).reset_index(drop=True)

# 7. Normal game scoring with AD/Deuce
score_map = ['0', '15', '30', '40']
normal_game_mask = ~merged['is_tiebreak']
tb_mask = merged['is_tiebreak']

# --- Normal games (vectorized) ---
p1 = merged.loc[normal_game_mask, 'p1_game_points_pre'].to_numpy()
p2 = merged.loc[normal_game_mask, 'p2_game_points_pre'].to_numpy()
servers = merged.loc[normal_game_mask, 'PointServer'].to_numpy()

scores = []
p1_scores = []
p2_scores = []

for a, b, s in zip(p1, p2, servers):
    if a >= 3 and b >= 3:
        if a == b:
            sc1, sc2 = '40', '40'
        elif a == b + 1:
            sc1, sc2 = 'AD', '40'
        elif b == a + 1:
            sc1, sc2 = '40', 'AD'
        else:
            sc1, sc2 = '40', '40'
    else:
        sc1, sc2 = score_map[min(a, 3)], score_map[min(b, 3)]
    
    # Swap if server is player 2
    if s == 2:
        sc1, sc2 = sc2, sc1
    
    scores.append(f"{sc1}-{sc2}")
    p1_scores.append(sc1)
    p2_scores.append(sc2)

merged.loc[normal_game_mask, 'score'] = scores
merged.loc[normal_game_mask, 'P1Score_Pre'] = p1_scores
merged.loc[normal_game_mask, 'P2Score_Pre'] = p2_scores

# --- Tiebreak games (vectorized) ---
p1_tb = merged.loc[tb_mask, 'p1_point'].groupby([merged['match_id'], merged['SetNo'], merged['GameNo']]).cumsum().shift(1).fillna(0).astype(int)
p2_tb = merged.loc[tb_mask, 'p2_point'].groupby([merged['match_id'], merged['SetNo'], merged['GameNo']]).cumsum().shift(1).fillna(0).astype(int)
servers_tb = merged.loc[tb_mask, 'PointServer']

# Swap for server
p1_scores_tb = np.where(servers_tb==1, p1_tb, p2_tb)
p2_scores_tb = np.where(servers_tb==1, p2_tb, p1_tb)

merged.loc[tb_mask, 'P1Score_Pre'] = p1_scores_tb.astype(str)
merged.loc[tb_mask, 'P2Score_Pre'] = p2_scores_tb.astype(str)
merged.loc[tb_mask, 'score'] = merged.loc[tb_mask, 'P1Score_Pre'] + '-' + merged.loc[tb_mask, 'P2Score_Pre']

# -----------------------------
# Server/Returner labeling
# -----------------------------
merged['server_point_win'] = merged['PointServer'] == merged['PointWinner']
merged['server_name'] = np.where(merged['PointServer']==1, merged['player1'], merged['player2'])
merged['returner_name'] = np.where(merged['PointServer']==1, merged['player2'], merged['player1'])

merged['server_name'] = merged['server_name'].apply(normalize_name)
merged['returner_name'] = merged['returner_name'].apply(normalize_name)
merged['player1'] = merged['player1'].apply(normalize_name)
merged['player2'] = merged['player2'].apply(normalize_name)

# -----------------------------
# Player stats
# -----------------------------
serve_stats = merged.groupby('server_name').agg(
    serve_points_won=('server_point_win', 'sum'),
    serve_points_total=('server_point_win', 'count')
).reset_index()
serve_stats['serve_point_win_pct'] = serve_stats['serve_points_won'] / serve_stats['serve_points_total']

merged['return_point_win'] = ~merged['server_point_win']
return_stats = merged.groupby('returner_name').agg(
    return_points_won=('return_point_win', 'sum'),
    return_points_total=('return_point_win', 'count')
).reset_index()
return_stats['return_point_win_pct'] = return_stats['return_points_won'] / return_stats['return_points_total']

player_stats = pd.merge(
    serve_stats[['server_name', 'serve_point_win_pct']],
    return_stats[['returner_name', 'return_point_win_pct']],
    left_on='server_name', right_on='returner_name', how='outer'
).rename(columns={'server_name': 'player'})
player_stats['serve_point_win_pct'] = player_stats['serve_point_win_pct'].fillna(0.62)
player_stats['return_point_win_pct'] = player_stats['return_point_win_pct'].fillna(0.38)
player_stats = player_stats[['player', 'serve_point_win_pct', 'return_point_win_pct']]

merged = merged.merge(
    player_stats.rename(columns={
        'player': 'player1',
        'serve_point_win_pct': 'player1_serve_point_win_pct',
        'return_point_win_pct': 'player1_return_point_win_pct'
    }),
    on='player1', how='left'
)
merged = merged.merge(
    player_stats.rename(columns={
        'player': 'player2',
        'serve_point_win_pct': 'player2_serve_point_win_pct',
        'return_point_win_pct': 'player2_return_point_win_pct'
    }),
    on='player2', how='left'
)

merged['player1_serve_point_win_pct'] = merged['player1_serve_point_win_pct'].fillna(0.62)
merged['player1_return_point_win_pct'] = merged['player1_return_point_win_pct'].fillna(0.38)
merged['player2_serve_point_win_pct'] = merged['player2_serve_point_win_pct'].fillna(0.62)
merged['player2_return_point_win_pct'] = merged['player2_return_point_win_pct'].fillna(0.38)

# -----------------------------
# Next game & last point of game
# -----------------------------
merged['next_GameNo'] = merged.groupby(['match_id', 'SetNo'])['GameNo'].shift(-1)
merged['is_last_point_of_game'] = merged['GameNo'] != merged['next_GameNo']
merged['GameWin'] = np.where(merged['is_last_point_of_game'], merged['PointWinner'], np.nan)
merged.loc[merged['is_tiebreak'], 'GameWin'] = merged.groupby(['match_id','SetNo','GameNo'])['PointWinner'].transform('last')
merged.drop(columns=['next_GameNo','is_last_point_of_game'], inplace=True)

# -----------------------------
# Best-of and round
# -----------------------------
merged[['best_of_5','round']] = merged['match_id'].apply(parse_match_id)

# -----------------------------
# Match winner
# -----------------------------
match_winners = merged.groupby('match_id').tail(1).copy()
match_winners['match_winner'] = np.where(match_winners['PointWinner']==1, match_winners['player1'], match_winners['player2'])
merged = merged.merge(match_winners[['match_id','match_winner']], on='match_id', how='left')

# -----------------------------
# Save CSV
# -----------------------------
keep_cols = ['match_id','SetNo','GameNo','PointNumber','PointWinner','PointServer','P1Score','P2Score','tournament','year',
             'player1','player2','P1_Sets_Won','P2_Sets_Won','P1_Games_Won','P2_Games_Won','score','P1Score_Pre','P2Score_Pre',
             'server_point_win','server_name','returner_name','player1_serve_point_win_pct','player1_return_point_win_pct',
             'player2_serve_point_win_pct','player2_return_point_win_pct','is_tiebreak','GameWin','best_of_5','round','match_winner'
]
# Automatically add all set columns from set_scores_wide
set_cols = [col for col in merged.columns if col.startswith("set")]
keep_cols += set_cols

merged = merged[keep_cols]

os.makedirs("data/processed", exist_ok=True)
# Detect int/float columns automatically
# Detect column types
int_cols = merged.select_dtypes(include=['int64','Int64']).columns.tolist()
float_cols = merged.select_dtypes(include=['float64']).columns.tolist()
str_cols = merged.select_dtypes(include=['object']).columns.tolist()

# 1. Trim strings and convert empty/whitespace-only strings to a placeholder
for c in str_cols:
    merged[c] = merged[c].astype(str).str.strip()           # trim whitespace
    merged[c] = merged[c].replace({"": "UNKNOWN", "nan": "UNKNOWN", "NaN": "UNKNOWN", "None": "UNKNOWN"})

# 2. Fill numeric columns
merged[int_cols] = merged[int_cols].fillna(0).astype(int)
merged[float_cols] = merged[float_cols].fillna(0.0)

# 3. Ensure all new computed columns are safe
# For example, score columns
for c in ['score','P1Score_Pre','P2Score_Pre']:
    if c in merged.columns:
        merged[c] = merged[c].fillna("0-0")

# 4. Sanity check
print("Nulls after cleaning:")
print(merged[int_cols].isna().sum())
print(merged[float_cols].isna().sum())
print(merged[str_cols].isna().sum())

print(merged.dtypes)
for c in merged.columns:
    print(c, merged[c].apply(type).value_counts().to_dict())

# 5. Save CSV
merged.to_csv("data/processed/merged_tennis_data.csv", index=False)
print("Saved cleaned merged_tennis_data.csv")
