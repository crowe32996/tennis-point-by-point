import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def compute_player_deltas(df):
    p1_df = df[["player1", "p1_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player1": "player", "p1_wp_delta": "wp_delta"}
    )
    p2_df = df[["player2", "p2_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player2": "player", "p2_wp_delta": "wp_delta"}
    )

    delta_df = pd.concat([p1_df, p2_df], ignore_index=True)

    high_pressure_df = delta_df[delta_df["is_high_pressure"]]

    # Aggregate deltas
    player_delta_summary_all = (
        delta_df.groupby("player")
        .agg(
            Total_Delta=("wp_delta", "sum"),
            Avg_Delta=("wp_delta", "mean"),
            Total_Points=("wp_delta", "count"),
            Tour=("Tour", lambda x: x.mode()[0] if not x.mode().empty else "Unknown")  # pick most common Tour
        )
        .reset_index()
    )

    player_delta_summary_hp = (
        high_pressure_df.groupby("player")
        .agg(
            Total_Delta_HP=("wp_delta", "sum"),
            Avg_Delta_HP=("wp_delta", "mean"),
            HP_Points=("wp_delta", "count")
        )
        .reset_index()
    )

    # Merge all together
    player_delta_summary = player_delta_summary_all.merge(
        player_delta_summary_hp, on="player", how="left"
    ).fillna(0)

    return player_delta_summary

@st.cache_data
def compute_match_player_consistency(df):
    """
    Compute per-match, per-player consistency metric.
    Returns a dataframe with:
        - match_id
        - player
        - wp_delta_std (std dev of win probability deltas)
        - consistency_score (inverse of std, higher = more consistent)
        - tourney_code (for future filtering/display)
        - year (optional)
        - Tour (optional)
    """
    results = []

    for match_id, group in df.groupby("match_id"):
        players = [group["player1"].iloc[0], group["player2"].iloc[0]]

        # grab match-level info
        tourney_code = group["tourney_code"].iloc[0] if "tourney_code" in group.columns else None

        for player in players:
            group_copy = group.copy()
            group_copy["player_wp_delta"] = group_copy.apply(
                lambda row: row["p1_wp_delta"] if player == row["player1"] else row["p2_wp_delta"], axis=1
            )

            wp_delta_std = group_copy["player_wp_delta"].std(ddof=0)  # population std
            consistency_score = 1 / wp_delta_std if wp_delta_std > 0 else None

            results.append({
                "match_id": match_id,
                "player": player,
                "wp_delta_std": wp_delta_std,
                "consistency_score": consistency_score,
                "tourney_code": tourney_code,
            })

    return pd.DataFrame(results)

@st.cache_data
def compute_clutch_rankings(df, min_hp_points = 200):
    # define server/return win columns
    df_local = df

    def compute_for(player_col, win_col):
        def summarize(group):
            all_points = group[win_col]
            hp_points = group.loc[group["is_high_pressure"], win_col]
            return pd.Series({
                "Win % (All)": all_points.mean(),
                "Win % (HP)": hp_points.mean() if len(hp_points) > 0 else float("nan"),
                "Total Points": len(all_points),
                "HP Points": len(hp_points),
                "Clutch Delta": (hp_points.mean() - all_points.mean()) if len(hp_points) > 0 else float("nan"),
                "Tour": group["tour"].iloc[0]  # take first Tour in the group
            })
        summary = df_local.groupby(player_col, group_keys=False).apply(summarize).dropna()
        summary = summary[summary["HP Points"] >= min_hp_points]
        return summary

    serve_stats = compute_for("server_name", "server_win")
    return_stats = compute_for("returner_name", "returner_win")
    stats = pd.concat([serve_stats, return_stats])
    stats = stats.groupby(stats.index).agg({
        "Win % (All)": "mean",
        "Win % (HP)": "mean",
        "Total Points": "sum",
        "HP Points": "sum",
        "Clutch Delta": "mean",
        "Tour": "first"  # take first Tour, or adjust if multiple
    })
    return stats

@st.cache_data
def compute_high_pressure_pct(df, min_hp_points):
    """
    Returns a dataframe with one row per player, containing:
    - Total Points
    - High Pressure Points
    - High Pressure % of total points
    """
    # Combine p1/p2 into a long format
    p1 = df[["player1", "is_high_pressure"]].rename(columns={"player1": "player"})
    p2 = df[["player2", "is_high_pressure"]].rename(columns={"player2": "player"})
    all_points = pd.concat([p1, p2], ignore_index=True)

    summary = all_points.groupby("player").agg(
        Total_Points=("is_high_pressure", "count"),
        High_Pressure_Points=("is_high_pressure", "sum")
    ).reset_index()

    summary["High_Pressure_Pct"] = summary["High_Pressure_Points"] / summary["Total_Points"]
    summary = summary[summary["High_Pressure_Points"] >= min_hp_points]
    return summary

@st.cache_data
def compute_top_points(df, top_n=10):
    df_local = df
    
    # True swing: probability if point won minus probability if point lost
    df_local["wp_delta_display"] = df_local["p1_win_prob_if_p1_wins"] - df_local["p1_win_prob_if_p2_wins"]
    df_local["abs_wp_delta"] = df_local["wp_delta_display"].abs()
    df_local["PointWinnerName"] = df_local.apply(
        lambda row: row["player1"] if row["point_winner"] == 1 else row["player2"],
        axis=1
    )
    top_points = df_local.nlargest(top_n, "abs_wp_delta")
    
    top_points[["year", "tourney_code", "match_num"]] = top_points["match_id"].str.split("-", n=2, expand=True)
    return top_points

@st.cache_data
def compute_unlikely_matches(df, low_threshold=0.10, high_threshold=0.90):
    # For unlikely winners & unlikely losers counts (one per match, per player)
    df_valid = df[df["match_winner"].notna()]
    df_valid = filter_matches_by_sets(df_valid)

    # Build p1/p2 version
    p1_df = df_valid
    p1_df["win_prob"] = p1_df["p1_win_prob_before"]
    p1_df["is_winner"] = p1_df["match_winner"] == p1_df["player1"]
    p1_df["player"] = p1_df["player1"]

    p2_df = df_valid
    p2_df["win_prob"] = 1 - p2_df["p1_win_prob_before"]
    p2_df["is_winner"] = p2_df["match_winner"] == p2_df["player2"]
    p2_df["player"] = p2_df["player2"]

    all_points = pd.concat([p1_df, p2_df], ignore_index=True)

    # Unlikely wins
    unlikely_wins = all_points[all_points["is_winner"] & (all_points["win_prob"] <= low_threshold)]
    unlikely_wins_unique = unlikely_wins.sort_values("win_prob").groupby(["player", "match_id"]).first().reset_index()
    wins_summary = (
        unlikely_wins_unique.groupby("player")["match_id"].nunique().reset_index().rename(columns={"match_id": "Num_UnlikelyWins"})
    ).sort_values("Num_UnlikelyWins", ascending=False)

    # Unlikely losses
    unlikely_losses = all_points[(~all_points["is_winner"]) & (all_points["win_prob"] >= high_threshold)]
    unlikely_losses_unique = unlikely_losses.sort_values("win_prob", ascending=False).groupby(["player", "match_id"]).first().reset_index()
    losses_summary = (
        unlikely_losses_unique.groupby("player")["match_id"].nunique().reset_index().rename(columns={"match_id": "Num_UnlikelyLosses"})
    ).sort_values("Num_UnlikelyLosses", ascending=False)

    return wins_summary, losses_summary


@st.cache_data
def compute_score_summary(df):
    score_df = df[df["score"].notna()]
    valid_scores = ["0", "15", "30", "40", "AD"]
    
    # Extract score components
    score_df[["server_score_val", "returner_score_val"]] = score_df["score"].str.split("-", expand=True)
    score_df = score_df[
        score_df["server_score_val"].isin(valid_scores) &
        score_df["returner_score_val"].isin(valid_scores)
    ]

    # Compute binary win column (Player 1 or server)
    score_df["point_win"] = (score_df["PointWinner"] == score_df["PointServer"]).astype(int)

    # Group by full score and compute summary
    score_summary = (
        score_df.groupby("score")
        .agg(
            Average_Importance=("importance", "mean"),
            Win_Rate=("point_win", "mean"),  # mean gives fraction of points won
            Points=("importance", "count")
        )
        .sort_values("Points", ascending=False)
        .reset_index()
    )
    return score_summary

@st.cache_data
def compute_match_player_clutch(df):
    """
    Compute per-match, per-player clutch score.
    Returns a dataframe with:
        - match_id
        - player
        - total_clutch_score
        - high_pressure_points
    """
    df_long = pd.concat([
        df.assign(player=df['player1'],
                  player_wp_delta=df['p1_wp_delta'],
                  points_stake=df['points_stake'],
                  importance=df['importance']),
                  #,is_high_pressure=df['is_high_pressure']),
        df.assign(player=df['player2'],
                  player_wp_delta=df['p2_wp_delta'],
                  points_stake=df['points_stake'],
                  importance=df['importance'])
                  #,is_high_pressure=df['is_high_pressure'])
    ], ignore_index=True)

    # Compute clutch score per point (vectorized)
    df_long['clutch_score'] = df_long['player_wp_delta'] * df_long['importance'] * df_long['points_stake']

    # Keep only necessary columns for aggregation
    df_long = df_long[['match_id', 'player', 'clutch_score']]

    # Aggregate per match, per player
    results = df_long.groupby(['match_id', 'player'], observed=True).agg(
        Total_Clutch_Score=('clutch_score', 'sum')
        #,High_Pressure_Points=('is_high_pressure', 'sum')
    ).reset_index()

    return results

@st.cache_data
def compute_player_clutch_aggregate(match_clutch_df):
    """
    Aggregate clutch score across all matches per player.
    """
    return match_clutch_df.groupby("player").agg(
        Total_Clutch_Score=("Total_Clutch_Score", "sum"),
        High_Pressure_Points=("High_Pressure_Points", "sum"),
        Avg_Clutch_Score=("Total_Clutch_Score", "mean")  # average per match
    ).reset_index()

