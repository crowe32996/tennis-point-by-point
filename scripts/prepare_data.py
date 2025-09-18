import pandas as pd
import numpy as np
import os
import requests
import re


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


# Grand Slam slugs and years
slams = ['ausopen', 'frenchopen', 'wimbledon', 'usopen']
years = list(range(2020, 2025))

points_df_list = []
matches_df_list = []

# Load all point-by-point and match data
for year in years:
    for slam in slams:
        points_file = f"{year}-{slam}-points.csv"
        matches_file = f"{year}-{slam}-matches.csv"
        points_path = f"data/raw/{points_file}"
        matches_path = f"data/raw/{matches_file}"

        base_url = "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"

        try:
            # Try downloading if not already present
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

# Merge on match_id
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

# ---- Score Reconstruction ----

score_values = ['0', '15', '30', '40']

def track_match_state(df):
    df = df.sort_values(['match_id', 'SetNo', 'GameNo', 'PointNumber']).copy()

    current_match_id = None
    prev_set_no = prev_game_no = None
    p1_sets = p2_sets = 0
    p1_games = p2_games = 0

    sets_p1, sets_p2, games_p1, games_p2 = [], [], [], []
    last_point_winner = None

    for idx, row in df.iterrows():
        match_id = row['match_id']
        set_no = row['SetNo']
        game_no = row['GameNo']
        point_winner = row['PointWinner']

        if match_id != current_match_id:
            current_match_id = match_id
            p1_sets = p2_sets = p1_games = p2_games = 0
            prev_set_no = prev_game_no = None

        if prev_set_no is not None:
            # New set
            if set_no != prev_set_no:
                # Add previous game's winner to game count
                if last_point_winner == 1:
                    p1_games += 1
                elif last_point_winner == 2:
                    p2_games += 1

                # Award set if needed
                if (p1_games >= 6 and p1_games - p2_games >= 2) or (p1_games == 7 and p2_games == 6):
                    p1_sets += 1
                elif (p2_games >= 6 and p2_games - p1_games >= 2) or (p2_games == 7 and p1_games == 6):
                    p2_sets += 1

                # Reset for new set
                p1_games = p2_games = 0

            # New game (still within same set)
            elif game_no != prev_game_no:
                if last_point_winner == 1:
                    p1_games += 1
                elif last_point_winner == 2:
                    p2_games += 1

                # Check if set should be awarded
                if (p1_games >= 6 and p1_games - p2_games >= 2) or (p1_games == 7 and p2_games == 6):
                    p1_sets += 1
                    p1_games = p2_games = 0
                elif (p2_games >= 6 and p2_games - p1_games >= 2) or (p2_games == 7 and p1_games == 6):
                    p2_sets += 1
                    p1_games = p2_games = 0

        sets_p1.append(p1_sets)
        sets_p2.append(p2_sets)
        games_p1.append(p1_games)
        games_p2.append(p2_games)

        prev_set_no = set_no
        prev_game_no = game_no
        last_point_winner = point_winner

    df['P1_Sets_Won'] = sets_p1
    df['P2_Sets_Won'] = sets_p2
    df['P1_Games_Won'] = games_p1
    df['P2_Games_Won'] = games_p2
    return df

def reconstruct_scores(df):
    df = df.sort_values(['match_id', 'SetNo', 'GameNo', 'PointNumber']).copy()
    scores, server_scores, returner_scores = [], [], []
    p1_scores, p2_scores = [], []

    p1_points = p2_points = 0
    p1_adv = p2_adv = False
    current_match_id = None
    prev_set_no = prev_game_no = None
    p1_games = p2_games = 0

    score_values = {0: "0", 1: "15", 2: "30", 3: "40", 4: "AD"}

    for _, row in df.iterrows():
        match_id, set_no, game_no, winner, server = row['match_id'], row['SetNo'], row['GameNo'], row['PointWinner'], row['PointServer']

        # Reset at new match
        if match_id != current_match_id:
            current_match_id = match_id
            p1_points = p2_points = 0
            p1_adv = p2_adv = False
            p1_games = p2_games = 0
            prev_set_no = prev_game_no = None

        # Reset at new set
        if prev_set_no is not None and set_no != prev_set_no:
            p1_points = p2_points = 0
            p1_adv = p2_adv = False
            p1_games = p2_games = 0

        if prev_game_no is not None and game_no != prev_game_no:
            p1_points = p2_points = 0
            p1_adv = p2_adv = False

        # ---- RECORD PRE-POINT SCORE ----
        if server == 1:
            server_score_str = score_values.get(min(p1_points, 3), "0") if not (p1_points >= 3 and p2_points >= 3 and p1_adv) else ("AD" if p1_adv else score_values.get(min(p1_points,3),"0"))
            returner_score_str = score_values.get(min(p2_points, 3), "0") if not (p1_points >= 3 and p2_points >= 3 and p2_adv) else ("AD" if p2_adv else score_values.get(min(p2_points,3),"0"))
        else:
            server_score_str = score_values.get(min(p2_points, 3), "0") if not (p1_points >= 3 and p2_points >= 3 and p2_adv) else ("AD" if p2_adv else score_values.get(min(p2_points,3),"0"))
            returner_score_str = score_values.get(min(p1_points, 3), "0") if not (p1_points >= 3 and p2_points >= 3 and p1_adv) else ("AD" if p1_adv else score_values.get(min(p1_points,3),"0"))

        scores.append(f"{server_score_str}-{returner_score_str}")
        server_scores.append(server_score_str)
        returner_scores.append(returner_score_str)
        p1_scores.append(score_values.get(min(p1_points, 3), "0"))
        p2_scores.append(score_values.get(min(p2_points, 3), "0"))

        # ---- UPDATE POINTS AFTER RECORDING ----
        in_tiebreak = (p1_games == 6 and p2_games == 6)

        if in_tiebreak:
            if winner == 1:
                p1_points += 1
            else:
                p2_points += 1
        else:
            if p1_points >= 3 and p2_points >= 3:
                if p1_adv:
                    if winner == 1:
                        p1_points = p2_points = 0
                        p1_adv = p2_adv = False
                    else:
                        p1_adv = False
                elif p2_adv:
                    if winner == 2:
                        p1_points = p2_points = 0
                        p1_adv = p2_adv = False
                    else:
                        p2_adv = False
                else:
                    if winner == 1:
                        p1_adv = True
                    else:
                        p2_adv = True
            else:
                if winner == 1:
                    p1_points += 1
                else:
                    p2_points += 1

                if p1_points >= 4 and (p1_points - p2_points) >= 2:
                    p1_points = p2_points = 0
                    p1_adv = p2_adv = False
                elif p2_points >= 4 and (p2_points - p1_points) >= 2:
                    p1_points = p2_points = 0
                    p1_adv = p2_adv = False

        prev_game_no = game_no
        prev_set_no = set_no

    df = df.copy()
    df['score'] = scores
    df['server_score_pre'] = server_scores
    df['returner_score_pre'] = returner_scores
    df['P1Score_Pre'] = p1_scores
    df['P2Score_Pre'] = p2_scores

    return df


def add_is_tiebreak_column(df):
    """
    Add a boolean 'is_tiebreak' column.
    """
    df['is_tiebreak'] = ((df['P1_Games_Won'] == 6) & (df['P2_Games_Won'] == 6))
    return df

def parse_match_id(match_id):
    match_id = str(match_id)
    
    # Check for MS/WS format
    m = re.search(r'-(MS|WS)(\d+)$', match_id)
    if m:
        best_of = 5 if m.group(1) == 'MS' else 3  # Men = 5 sets, Women = 3 sets
        round_num = int(m.group(2)[0])  # first digit after MS/WS
        return pd.Series([best_of, round_num])
    
    # Fallback for numeric IDs (e.g., 2020-ausopen-1201)
    m2 = re.search(r'(\d{4})$', match_id)
    if m2:
        last4 = m2.group(1)
        gender_digit = int(last4[0])  # 1 = men, 2 = women
        best_of = 5 if gender_digit == 1 else 3
        round_num = int(last4[1])  # second digit as round
        return pd.Series([best_of, round_num])
    
    # If nothing matches
    return pd.Series([np.nan, np.nan])

merged = merged.groupby('match_id', group_keys=False)\
               .apply(track_match_state).reset_index(drop=True)


merged = merged.groupby(['match_id', 'SetNo', 'GameNo'], group_keys=False)\
               .apply(lambda group: reconstruct_scores(group)).reset_index(drop=True)


keep_cols = ['match_id','SetNo','GameNo','PointNumber','PointWinner','PointServer',	'P1Score','P2Score','tournament','year','player1','player2',
             'P1_Sets_Won','P2_Sets_Won','P1_Games_Won','P2_Games_Won','score','P1Score_Pre','P2Score_Pre'
]

merged = merged[keep_cols]

# ---- End Score Reconstruction ----

# Label server and returner
merged['server_point_win'] = merged['PointServer'] == merged['PointWinner']
merged['server_name'] = merged.apply(lambda row: row['player1'] if row['PointServer'] == 1 else row['player2'], axis=1)
merged['returner_name'] = merged.apply(lambda row: row['player2'] if row['PointServer'] == 1 else row['player1'], axis=1)

def normalize_name(name):
    if pd.isna(name): return name
    parts = name.split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else name

merged['server_name'] = merged['server_name'].apply(normalize_name)
merged['returner_name'] = merged['returner_name'].apply(normalize_name)
merged['player1'] = merged['player1'].apply(normalize_name)
merged['player2'] = merged['player2'].apply(normalize_name)

# Aggregate serve and return stats
server_df = merged.groupby(['server_name', 'score']).agg(
    serve_points_won=('server_point_win', 'sum'),
    serve_points_total=('server_point_win', 'count')
).reset_index().rename(columns={'server_name': 'player'})

return_df = merged.groupby(['returner_name', 'score']).agg(
    return_points_won=('server_point_win', lambda x: (~x).sum()),
    return_points_total=('server_point_win', 'count')
).reset_index().rename(columns={'returner_name': 'player'})

summary = pd.merge(server_df, return_df, on=['player', 'score'], how='outer').fillna(0)
summary['serve_win_pct'] = summary['serve_points_won'] / summary['serve_points_total'].replace(0, np.nan)
summary['return_win_pct'] = summary['return_points_won'] / summary['return_points_total'].replace(0, np.nan)

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

merged['next_GameNo'] = merged.groupby(['match_id', 'SetNo'])['GameNo'].shift(-1)
merged['is_last_point_of_game'] = merged['GameNo'] != merged['next_GameNo']
# Standard game winner
merged['GameWin'] = np.where(merged['is_last_point_of_game'], merged['PointWinner'], np.nan)

# Step 3: Add the is_tiebreak column
merged = add_is_tiebreak_column(merged)

# Correct for tiebreaks: last point of the tiebreak wins the game
merged.loc[merged['is_tiebreak'], 'GameWin'] = merged.groupby(['match_id','SetNo','GameNo'])['PointWinner'].transform('last')
merged.drop(columns=['next_GameNo', 'is_last_point_of_game'], inplace=True)


def overwrite_tiebreak_scores(df):
    df = df.copy()
    p1_tb = 0
    p2_tb = 0
    current_game = None

    for i, row in df.iterrows():
        key = (row['match_id'], row['SetNo'], row['GameNo'])
        if current_game != key:
            p1_tb = 0
            p2_tb = 0
            current_game = key
        
        if row['PointWinner'] == 1:
            p1_tb += 1
        else:
            p2_tb += 1

        df.at[i, 'P1Score_Pre'] = str(p1_tb)
        df.at[i, 'P2Score_Pre'] = str(p2_tb)
        df.at[i, 'score'] = f"{p1_tb}-{p2_tb}"  # note the single quote at the beginning
    
    return df

# Separate tiebreak and non-tiebreak rows
tiebreak_rows = merged[merged['is_tiebreak']].copy()
non_tiebreak_rows = merged[~merged['is_tiebreak']].copy()

# Only update scores for tiebreak rows
tiebreak_rows = overwrite_tiebreak_scores(tiebreak_rows)

# Combine back together and sort
merged = pd.concat([non_tiebreak_rows, tiebreak_rows]).sort_values(by=['match_id', 'SetNo', 'GameNo', 'PointNumber']).reset_index(drop=True)

# Apply best_of_5 and round columns to the dataframe
merged[['best_of_5', 'round']] = merged['match_id'].apply(parse_match_id)

# --- Add match winner ---
# Take the last point of each match to determine who won more sets
match_winners = merged.groupby('match_id').tail(1).copy()
match_winners['match_winner'] = np.where(
    match_winners['PointWinner'] == 1,
    match_winners['player1'],
    match_winners['player2']
)

# Merge back the match winner to the full dataframe
merged = merged.merge(
    match_winners[['match_id', 'match_winner']],
    on='match_id',
    how='left'
)

# Save to CSV
merged.to_csv("data/processed/merged_tennis_data.csv", index=False)
print("Saved merged_tennis_data.csv")