# streamlit_app.py
import streamlit as st
import pandas as pd
import duckdb
import os
import altair as alt
from pathlib import Path
import numpy as np
import altair as alt
from streamlit.components.v1 import html as components_html
import psutil


# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"
CSV_FILE = BASE_DIR / "outputs" / "all_points_with_importance.csv"
TABLE_NAME = "importance_results"
FEATURE_COLUMNS = {
    "base": ["match_id", "player1", "player2", "year"],  # always needed
    "win_prob": ["p1_win_prob_before", 'p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins', "p1_wp_delta", "p2_wp_delta", "points_stake","importance"],
    "serve_return": ["PointWinner", "PointServer", "server_name", "returner_name"],
    "match_winner": ["match_winner"],
    "score": ["P1_Sets_Won", "P2_Sets_Won", "P1_Games_Won", "P2_Games_Won", "score"],
}


PLAYER_COUNTRY_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"
player_country_df = pd.read_csv(PLAYER_COUNTRY_FILE)
player_flag_map = dict(zip(player_country_df["player"], player_country_df["country"]))

def print_memory(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2  # Resident Set Size in MB
    st.text(f"[MEMORY] {note} - {mem_mb:.2f} MB")


@st.cache_data
def load_filtered_df_sql(selected_years, selected_tour="All", selected_tourney="All",
                         selected_players="All", min_points_filter=None, columns=None):
    """
    Load DuckDB data with all filters applied in SQL.
    Total points filter is applied per player, not sets.
    Optimized to calculate player totals and active players only once.
    """
    if columns is None:
        columns = FEATURE_COLUMNS["base"] + FEATURE_COLUMNS["win_prob"] + FEATURE_COLUMNS["serve_return"]

    essential_cols = ["match_id", "player1", "player2", "year", 
                      "PointWinner", "p1_win_prob_if_p1_wins", 
                      "p1_win_prob_if_p2_wins", "p1_win_prob_before"]

    all_cols = list(dict.fromkeys(essential_cols + columns))
    cols_str = ", ".join(all_cols)
    years_str = ",".join(map(str, selected_years))

    con = duckdb.connect(DUCKDB_FILE)

    most_recent_year = con.execute(
        f"SELECT MAX(year) FROM importance_results WHERE year IN ({years_str})"
    ).fetchone()[0]

    # Build optional min points filter clause
    min_points_filter_sql = ""
    if min_points_filter:
        min_points_filter_sql = f"""
        JOIN player_totals pt1 ON b.player1 = pt1.player
        LEFT JOIN player_totals pt2 ON b.player2 = pt2.player
        WHERE (pt1.player IS NOT NULL OR pt2.player IS NOT NULL)
        """
    
    # Build active/inactive player filter
    player_filter_sql = ""
    if selected_players == "Active":
        player_filter_sql = f"""
        AND (b.player1 IN (SELECT player FROM active_players)
             OR b.player2 IN (SELECT player FROM active_players))
        """
    elif selected_players == "Inactive":
        player_filter_sql = f"""
        AND (b.player1 NOT IN (SELECT player FROM active_players)
             OR b.player2 NOT IN (SELECT player FROM active_players))
        """

    query = f"""
    WITH base AS (
        SELECT
            {cols_str},
            CASE
                WHEN match_id LIKE '%-MS%' THEN 'ATP'
                WHEN match_id LIKE '%-WS%' THEN 'WTA'
                WHEN LENGTH(match_id) >= 4 AND substr(match_id, -4, 1) = '1' THEN 'ATP'
                WHEN LENGTH(match_id) >= 4 AND substr(match_id, -4, 1) = '2' THEN 'WTA'
                ELSE 'Unknown'
            END AS Tour,
            split_part(match_id, '-', 2) AS tourney_code,
            CASE
                WHEN PointWinner = 1 THEN p1_win_prob_if_p1_wins - p1_win_prob_before
                ELSE p1_win_prob_if_p2_wins - p1_win_prob_before
            END AS p1_wp_delta,
            CASE
                WHEN PointWinner = 1 THEN p1_win_prob_before - p1_win_prob_if_p1_wins
                ELSE p1_win_prob_before - p1_win_prob_if_p2_wins
            END AS p2_wp_delta
        FROM importance_results
        WHERE year IN ({years_str})
    ),
    player_totals AS (
        SELECT player, SUM(total_points) AS total_points
        FROM (
            SELECT player1 AS player, COUNT(*) AS total_points
            FROM importance_results
            WHERE year IN ({years_str})
            GROUP BY player1
            UNION ALL
            SELECT player2 AS player, COUNT(*) AS total_points
            FROM importance_results
            WHERE year IN ({years_str})
            GROUP BY player2
        )
        GROUP BY player
        HAVING SUM(total_points) >= {min_points_filter if min_points_filter else 0}
    ),
    active_players AS (
        SELECT player1 AS player FROM importance_results WHERE year={most_recent_year}
        UNION
        SELECT player2 AS player FROM importance_results WHERE year={most_recent_year}
    )
    SELECT b.*
    FROM base b
    LEFT JOIN player_totals pt1 ON b.player1 = pt1.player
    LEFT JOIN player_totals pt2 ON b.player2 = pt2.player
    WHERE (b.Tour = '{selected_tour}' OR '{selected_tour}'='All')
      AND (b.tourney_code = '{selected_tourney}' OR '{selected_tourney}'='All')
      {"AND (pt1.player IS NOT NULL OR pt2.player IS NOT NULL)" if min_points_filter else ""}
      {player_filter_sql}
    """

    # st.text("=== DuckDB Query ===")
    # st.code(query, language="sql")
    # st.text("===================")

    df = con.execute(query).fetchdf()
    con.close()

    # Clean player names and map tournaments
    df["player1"] = df["player1"].map(clean_player_name)
    df["player2"] = df["player2"].map(clean_player_name)
    df["tournament"] = df["tourney_code"].map(TOURNAMENTS_MAP).fillna(df["tourney_code"])

    return df


@st.cache_data
def load_df_from_duckdb(selected_years, selected_tour, columns):
    """
    Load DuckDB data with the specified columns and years, and derive Tour from match_id.
    Handles:
      - standard match_id: 2023-usopen-1101 (4th-to-last digit: 1=Men, 2=Women)
      - alternate: 2023-ausopen-MS101 (MS=Men, WS=Women)
    """
    # Always include base columns
    columns_to_select = list(dict.fromkeys(FEATURE_COLUMNS["base"] + columns))
    cols_str = ", ".join(columns_to_select)
    
    # Build the year filter
    years_str = ",".join(map(str, selected_years))
    
    # Query DuckDB
    query = f"""
        SELECT {cols_str}
        FROM importance_results
        WHERE year IN ({years_str})
    """
    
    con = duckdb.connect(DUCKDB_FILE)
    df = con.execute(query).fetchdf()
    con.close()
    
    # --- Derive Tour from match_id ---
    def extract_gender(match_id: str) -> str:
        match_id_str = str(match_id)
        if "-" in match_id_str:
            last_part = match_id_str.split("-")[-1]
            if last_part.startswith("MS"):
                return "Men"
            elif last_part.startswith("WS"):
                return "Women"
        if len(match_id_str) >= 4:
            gender_digit = match_id_str[-4]
            if gender_digit == "1":
                return "Men"
            elif gender_digit == "2":
                return "Women"
        return None

    if "match_id" in df.columns:
        df["gender"] = df["match_id"].apply(extract_gender)
        df["Tour"] = df["gender"].map({"Men": "ATP", "Women": "WTA"})
    else:
        df["Tour"] = "Unknown"
    
    # Filter by tour if not "All"
    if selected_tour != "All":
        df = df[df["Tour"] == selected_tour]
    
    return df


def clean_player_name(name: str) -> str:
    """Capitalize player names correctly, handling spaces, hyphens, and apostrophes."""
    if not name or not name.strip():
        return name

    parts = name.split()
    first, rest_parts = parts[0], parts[1:]

    cleaned_rest = []
    for word in rest_parts:
        for sep in ["-", "'"]:
            if sep in word:
                word = sep.join([w.capitalize() for w in word.split(sep)])
        cleaned_rest.append(word.capitalize() if all(sep not in word for sep in ["-", "'"]) else word)

    return f"{first} {' '.join(cleaned_rest)}".strip()

def add_basic_columns(df):
    # --- win probability deltas (vectorized, faster than apply) ---
    if "p1_wp_delta" not in df.columns and all(col in df.columns for col in ["PointWinner", "p1_win_prob_if_p1_wins", "p1_win_prob_if_p2_wins", "p1_win_prob_before"]):
        df["p1_wp_delta"] = np.where(
            df["PointWinner"] == 1,
            df["p1_win_prob_if_p1_wins"] - df["p1_win_prob_before"],
            df["p1_win_prob_if_p2_wins"] - df["p1_win_prob_before"]
        )
        df["p2_wp_delta"] = -df["p1_wp_delta"]

    # --- player name cleanup ---
    if "player1" in df.columns and "player2" in df.columns:
        df["player1"] = df["player1"].map(clean_player_name)
        df["player2"] = df["player2"].map(clean_player_name)

    # --- gender + Tour extraction ---
    if "match_id" in df.columns:
        # --- Tournament extraction from match_id ---
        # Example match_id: 2023-ausopen-1120
        df["tourney_code"] = df["match_id"].astype(str).str.split("-", n=2).str[1]
        df["tournament"] = df["tourney_code"].map(TOURNAMENTS_MAP).fillna(df["tourney_code"])

    return df

def filter_matches_by_sets(df: pd.DataFrame) -> pd.DataFrame:
    """S
    Remove invalid matches from the dataset:
      - Men: exclude if a player has 3 sets won or more
      - Women: exclude if a player has 2 sets won or more
    """
    def is_valid(group):
        tour = group["Tour"].iloc[0]
        p1_sets = group["P1_Sets_Won"].max()
        p2_sets = group["P2_Sets_Won"].max()
        if tour in ("ATP",'All') and (p1_sets >= 3 or p2_sets >= 3):
            return False
        if tour == "WTA" and (p1_sets >= 2 or p2_sets >= 2):
            return False
        return True

    return df.groupby("match_id", group_keys=False).filter(is_valid).reset_index(drop=True)


def apply_filters(df, selected_tourney, selected_tour, selected_years):
    df2 = df
    df2["tournament"] = df2["tournament"].map(TOURNAMENTS_MAP).fillna(df2["tournament"])
    if selected_tourney != "All":
        df2 = df2[df2["tournament"] == selected_tourney]
    if selected_tour != "All":
        df2 = df2[df2["Tour"] == selected_tour]
    if selected_years:
        df2 = df2[df2["year"].isin(selected_years)]
    return df2

def add_filtered_player_columns(df, selected_players):
    most_recent_year = df['year'].max()
    active_players = pd.unique(
        df[df['year'] == most_recent_year][['player1', 'player2']].values.ravel()
    )
    inactive_players = [p for p in pd.unique(df[['player1', 'player2']].values.ravel())
                        if p not in active_players]

    def mask_player(player_name):
        if selected_players == "All":
            return player_name
        elif selected_players == "Active":
            return player_name if player_name in active_players else None
        elif selected_players == "Inactive":
            return player_name if player_name in inactive_players else None

    df['player1_filtered'] = df['player1'].apply(mask_player)
    df['player2_filtered'] = df['player2'].apply(mask_player)
    return df


IOC_TO_ISO2 = {
    "SUI": "CH",
    "DEN": "DK",
    "GER": "DE",
    "BLR": "BY",
    "POL": "PL",
    "TUN": "TN",
    "UKR": "UA",
    "BUL": "BG",
    "SRB": "RS",
    "KAZ": "KZ",
    "RSA": "ZA",
    "RUS": "RU",
    "CHI": "CN",
    "TPE": "TW",
    "PUR": "PR",
    "ESA": "SV",
    "INA": "ID",
    "VAN": "VU",
    "NMI": "MP",
    "POC": "XK",
    "IRI": "IR",
    "SWE": "SE",  
    "CHI": "CL",  
    "AUT": "AT",  
    "NED": "NL",
    "CRO": "HR"
}

def ioc_to_flag(ioc_code):
    iso2 = IOC_TO_ISO2.get(ioc_code, ioc_code[:2].upper())
    OFFSET = 127397
    return "".join([chr(ord(c) + OFFSET) for c in iso2])
    
def add_flag_to_player(df, player_col="Player"):
    df[player_col] = df[player_col].map(
        lambda x: f"{ioc_to_flag(player_flag_map.get(x, ''))} {x}" if player_flag_map.get(x) else x
    )
    return df

def add_flag_image(df, player_col="Player"):
    def flag_img_html(player_name):
        ioc = player_flag_map.get(player_name, "")
        if ioc:
            iso2 = IOC_TO_ISO2.get(ioc, ioc[:2].upper())
            url = f"https://flagcdn.com/w20/{iso2.lower()}.png"
            return f'<img src="{url}" width="20" style="vertical-align:middle;margin-right:4px">{player_name}'
        else:
            return player_name

    df[player_col] = df[player_col].apply(flag_img_html)
    return df

def render_flag_table(df, player_col="Player", numeric_cols=None, max_height=400):
    """
    Render a DataFrame with flags using HTML and a scrollable container.
    df: dataframe with player names and stats
    player_col: column with player names
    numeric_cols: list of numeric/stat columns to display
    max_height: max height of table in pixels (scrollbar appears if exceeded)
    """
    if numeric_cols is None:
        numeric_cols = [c for c in df.columns if c != player_col]

    html = '<div style="overflow-y:visible;">'
    html += '<table style="width:100%; border-collapse: collapse;">'
    html += "<tr><th style='text-align:left'>Player</th>"
    for col in numeric_cols:
        html += f"<th style='text-align:right'>{col}</th>"
    html += "</tr>"

    for _, row in df.iterrows():
        player_name = row[player_col]
        ioc_code = player_flag_map.get(player_name, "")
        flag_html = ""
        if ioc_code:
            iso2 = IOC_TO_ISO2.get(ioc_code, ioc_code[:2].upper())
            url = f"https://flagcdn.com/w20/{iso2.lower()}.png"
            flag_html = f'<img src="{url}" width="20" style="vertical-align:middle;margin-right:4px">'
        html += "<tr>"
        html += f"<td>{flag_html}{player_name}</td>"
        for col in numeric_cols:
            val = row[col]
            if isinstance(val, float):
                if col in ["High Pressure %","Win % (High Pressure)"]:
                    html += f"<td style='text-align:right'>{val*100:.1f}%</td>"
                else:
                    html += f"<td style='text-align:right'>{val:.3f}</td>"
            else:
                html += f"<td style='text-align:right'>{val}</td>"
        html += "</tr>"
    html += "</table></div>"

    st.markdown(html, unsafe_allow_html=True)

@st.cache_data
def compute_player_deltas(df):
    p1_df = df[["player1_filtered", "p1_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player1_filtered": "player", "p1_wp_delta": "wp_delta"}
    )
    p2_df = df[["player2_filtered", "p2_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player2_filtered": "player", "p2_wp_delta": "wp_delta"}
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
        players = [group["player1_filtered"].iloc[0], group["player2_filtered"].iloc[0]]

        # grab match-level info
        tourney_code = group["tourney_code"].iloc[0] if "tourney_code" in group.columns else None

        for player in players:
            group_copy = group.copy()
            group_copy["player_wp_delta"] = group_copy.apply(
                lambda row: row["p1_wp_delta"] if player == row["player1_filtered"] else row["p2_wp_delta"], axis=1
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
                "Tour": group["Tour"].iloc[0]  # take first Tour in the group
            })
        summary = df_local.groupby(player_col).apply(summarize).dropna()
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
        lambda row: row["player1"] if row["PointWinner"] == 1 else row["player2"],
        axis=1
    )
    top_points = df_local.nlargest(top_n, "abs_wp_delta")
    
    top_points[["year", "tourney_code", "match_num"]] = top_points["match_id"].str.split("-", n=2, expand=True)
    top_points["tournament_name"] = top_points["tourney_code"].map(TOURNAMENTS_MAP).fillna(top_points["tourney_code"])
    return top_points

@st.cache_data
def compute_unlikely_matches(df, low_threshold=0.10, high_threshold=0.90):
    # For unlikely winners & unlikely losers counts (one per match, per player)
    df_valid = df[df["match_winner"].notna()]
    df_valid = filter_matches_by_sets(df_valid)

    # Build p1/p2 version
    p1_df = df_valid
    p1_df["win_prob"] = p1_df["p1_win_prob_before"]
    p1_df["is_winner"] = p1_df["match_winner"] == p1_df["player1_filtered"]
    p1_df["player"] = p1_df["player1_filtered"]

    p2_df = df_valid
    p2_df["win_prob"] = 1 - p2_df["p1_win_prob_before"]
    p2_df["is_winner"] = p2_df["match_winner"] == p2_df["player2_filtered"]
    p2_df["player"] = p2_df["player2_filtered"]

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
        df.assign(player=df['player1_filtered'],
                  player_wp_delta=df['p1_wp_delta'],
                  points_stake=df['points_stake'],
                  importance=df['importance']),
                  #,is_high_pressure=df['is_high_pressure']),
        df.assign(player=df['player2_filtered'],
                  player_wp_delta=df['p2_wp_delta'],
                  points_stake=df['points_stake'],
                  importance=df['importance'])
                  #,is_high_pressure=df['is_high_pressure'])
    ], ignore_index=True)

    # Compute clutch score per point (vectorized)
    df_long['clutch_score'] = df_long['player_wp_delta'] * df_long['importance'] * df_long['points_stake']

    # Keep only necessary columns for aggregation
    df_long = df_long[['match_id', 'player', 'clutch_score', #'is_high_pressure', 
                       'tourney_code']]

    # Aggregate per match, per player
    results = df_long.groupby(['match_id', 'player', 'tourney_code'], observed=True).agg(
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


def render_scoreboard(row, height = 130):
    flag_p1 = "🇪🇸" if "Nadal" in row["Player 1"] else ""
    flag_p2 = "🇷🇸" if "Djokovic" in row["Player 2"] else ""
    tournament_logo = "🎾"

    # Add serve emoji to serving player
    p1_name = f"{flag_p1} {row['Player 1']}"
    p2_name = f"{flag_p2} {row['Player 2']}"
    if str(row["Server"]) in ["1", "Player 1"]:
        p1_name += " 🎾"
        server_first = True
    else:
        p2_name += " 🎾"
        server_first = False

    # Bold winner
    if row["Match Winner"] == row["Player 1"]:
        p1_name = f"<strong>{p1_name}</strong>"
    else:
        p2_name = f"<strong>{p2_name}</strong>"

    # Split game score at hyphen
    if "-" in str(row["Game Score"]):
        score_parts = row["Game Score"].split("-", 1)
        score_p1 = score_parts[0] if server_first else score_parts[1]
        score_p2 = score_parts[1] if server_first else score_parts[0]
    else:
        score_p1 = row["Game Score"]
        score_p2 = row["Game Score"]

    html = f"""
    <div style="
        border: 2px solid #ddd; 
        border-radius: 12px; 
        padding: 4px; 
        margin-bottom: 2px; 
        box-shadow: 1px 1px 4px rgba(0,0,0,0.08);
        font-family: Arial, sans-serif;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
            <span style="font-weight: bold; font-size: 1em;">{row['Tournament']} {row['Year']}</span>
            <span style="font-size: 1.2em;">{tournament_logo}</span>
        </div>

        <table style="width:100%; text-align: center; border-collapse: collapse; font-size: 0.9em;">
            <tr>
                <th style="text-align:left;">Player</th>
                <th>Sets</th>
                <th>Games</th>
                <th>Game Score</th>
            </tr>
            <tr>
                <td style="text-align:left;">{p1_name}</td>
                <td>{row['P1 Sets']}</td>
                <td>{row['P1 Games']}</td>
                <td>{score_p1}</td>
            </tr>
            <tr>
                <td style="text-align:left;">{p2_name}</td>
                <td>{row['P2 Sets']}</td>
                <td>{row['P2 Games']}</td>
                <td>{score_p2}</td>
            </tr>
        </table>

        <div style="margin-top:2px; font-size:0.85em; color:#555; text-align:center;">
            Lowest Win Probability: {row['Win Probability']:.1f}%
        </div>
    </div>
    """

    # Render the HTML inside an iframe (set height so iframe matches content)
    components_html(html, height=height)

def make_score_heatmap(df):
    return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rect()
# --------------------------
# Streamlit UI
# --------------------------

TOURNAMENTS_MAP = {
    "ausopen": "Australian Open",
    "frenchopen": "French Open",
    "wimbledon": "Wimbledon",
    "usopen": "US Open"
}
TOURNAMENTS_SIDEBAR = ["All"] + list(TOURNAMENTS_MAP.values())

TOURS = ["ATP", "WTA"]
default_tour = "ATP"
PLAYERS = ["All", "Active", "Inactive"]
default_players = "Active"

st.set_page_config(layout="wide", page_title="Tennis Clutch Dashboard")
st.sidebar.header("Filters")


# --- At the very top of your script ---
if "last_filters" not in st.session_state:
    st.session_state.last_filters = {}
if "force_rerun" not in st.session_state:
    st.session_state.force_rerun = False

print_memory("before any major ops")


# ---- Sidebar Filters ----
all_years = list(range(2020, 2025))
selected_year_range = st.sidebar.select_slider(
    "Select Year Range",
    options=all_years,
    value=(2020, 2024)
)
selected_years = list(range(selected_year_range[0], selected_year_range[1] + 1))

selected_tour = st.sidebar.selectbox("Tour", TOURS, index=0, key="tour_select")

# Dynamically update title
st.title(f"🎾 {selected_tour} Tennis Clutch Performers")
st.markdown(
    f"Analyze which players have **thrived** or **struggled** under pressure in {selected_tour} Grand Slam matches since 2020."
)
selected_players = st.sidebar.selectbox("Player Status", PLAYERS, index=PLAYERS.index(default_players))
selected_tourney = st.sidebar.selectbox("Tournament", TOURNAMENTS_SIDEBAR, index=0)

# Compute default min points
default_min_points = (400 if selected_tour == "ATP" else 200) * len(selected_years)
min_points_filter = st.sidebar.slider(
    "Minimum Points per Player",
    min_value=0,
    max_value=5000,
    value=default_min_points,
    step=50
)

filters = {
    "years": selected_years,
    "tour": selected_tour,
    "tourney": selected_tourney,
    "players": selected_players,
    "min_points": min_points_filter
}
filters_changed = filters != st.session_state.last_filters
if filters_changed:
    st.session_state.last_filters = filters
    st.session_state.force_rerun = True

# --- Trigger rerun if needed ---
if st.session_state.force_rerun:
    st.session_state.force_rerun = False
    st.rerun()

# --------------------------
# Tabs
# --------------------------
tab0, tab1, tab2 = st.tabs(["Player Performance", "High Pressure Performance", "Standout Events"])
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "tab0"

selected_tab = st.session_state.active_tab

# ==========================
# TAB 0: Player Performance
# ==========================
with tab0:
    st.subheader("📊 Player Consistency and Clutchness")
    st.session_state.active_tab = "tab0"
    if "df_tab0" not in st.session_state or filters_changed:
        st.session_state.df_tab0 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
        st.session_state.df_tab0 = add_filtered_player_columns(st.session_state.df_tab0, selected_players)
    df_tab0 = st.session_state.df_tab0


    # # ---- Lazy-load only columns needed for Tab 0 ----
    # df_tab0 = load_df_from_duckdb(
    #     selected_years=selected_years,
    #     selected_tour=selected_tour,
    #     columns=FEATURE_COLUMNS["base"] + FEATURE_COLUMNS["win_prob"] + FEATURE_COLUMNS["serve_return"]
    # )
    # df_tab0 = add_basic_columns(df_tab0)
    # df_tab0 = apply_filters(df_tab0, selected_tourney, selected_tour, selected_years)
    # df_tab0 = add_filtered_player_columns(df_tab0, selected_players)
    # df_tab0 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
    # df_tab0 = add_filtered_player_columns(df_tab0, selected_players)
    print_memory("after pulling in tab0 df")

    # ---- Add derived columns ----
    df_tab0["server_point_win"] = df_tab0["PointWinner"] == df_tab0["PointServer"]
    df_tab0["server_win"] = df_tab0["server_point_win"]
    df_tab0["returner_win"] = ~df_tab0["server_point_win"]

    # ---- Compute clutch dataframe ----
    match_clutch_df = compute_match_player_clutch(df_tab0)

    total_clutch_df = (
        match_clutch_df.groupby("player", as_index=False)["Total_Clutch_Score"]
        .sum()
        .rename(columns={"player": "Player", "Total_Clutch_Score": "Expected Points Added (EPA)"})
    ) if not match_clutch_df.empty else pd.DataFrame(columns=["Player", "Expected Points Added (EPA)"])

    # ---- Flatten to long format for consistency calculations ----
    df_long = pd.concat([
        df_tab0.assign(
            player=df_tab0['player1_filtered'],
            player_wp_delta=df_tab0['p1_wp_delta'],
            wp_before_point=df_tab0['p1_win_prob_before']
        ),
        df_tab0.assign(
            player=df_tab0['player2_filtered'],
            player_wp_delta=df_tab0['p2_wp_delta'],
            wp_before_point=1 - df_tab0['p1_win_prob_before']
        )
    ], ignore_index=True)

    # ---- Aggregate per player ----
    player_consistency_df = df_long.groupby('player', observed=True).agg(
        Total_Points=('player_wp_delta', 'count'),
        Avg_WP=('wp_before_point', 'mean'),
        WP_Delta_Std=('player_wp_delta', 'std')
    ).reset_index()

    # ---- Consistency metrics ----
    player_consistency_df['Consistency'] = player_consistency_df.apply(
        lambda r: r['Avg_WP'] / r['WP_Delta_Std'] if r['WP_Delta_Std'] > 0 else None,
        axis=1
    )
    player_consistency_df['Weighted_Consistency'] = player_consistency_df['Consistency'] * np.sqrt(player_consistency_df['Total_Points'])
    player_consistency_df['Consistency_Percentile'] = (
        (player_consistency_df['Weighted_Consistency'] - player_consistency_df['Weighted_Consistency'].min())
        / (player_consistency_df['Weighted_Consistency'].max() - player_consistency_df['Weighted_Consistency'].min())
    )

    player_consistency_df = player_consistency_df.rename(columns={'player':'Player', 'Consistency_Percentile': 'Consistency Index'})
    player_consistency_df = player_consistency_df[player_consistency_df['Total_Points'] >= min_points_filter]

    # ---- Merge with Clutch ----
    player_stats_df = player_consistency_df.merge(
        total_clutch_df[['Player', 'Expected Points Added (EPA)']],
        on='Player',
        how='inner'
    )
    player_stats_df['Clutch_Percentile'] = player_stats_df['Expected Points Added (EPA)'].rank(pct=True)

    # ---- Bubble chart ----
    tennis_colors = alt.Scale(domain=[player_stats_df['Total_Points'].min(),
                                      player_stats_df['Total_Points'].max()],
                              range=["#ffffcc", "#ccff00"])

    bubble = alt.Chart(player_stats_df).mark_circle().encode(
        x=alt.X('Consistency Index', scale=alt.Scale(domain=[0,1])),
        y=alt.Y('Clutch_Percentile', scale=alt.Scale(domain=[0,1])),
        size=alt.Size('Expected Points Added (EPA)', scale=alt.Scale(range=[50, 1000])),
        color=alt.Color('Total_Points', scale=tennis_colors),
        tooltip=[
            alt.Tooltip('Player:N'),
            alt.Tooltip('Total_Points:Q', format=','),
            alt.Tooltip('Consistency Index:Q', format='.1%'),
            alt.Tooltip('Clutch_Percentile:Q', format='.1%'),
            alt.Tooltip('Expected Points Added (EPA):Q', format='.0f')
        ]
    )

    vline = alt.Chart(pd.DataFrame({'Consistency Index':[0.5]})).mark_rule(color='gray', strokeDash=[4,4]).encode(x='Consistency Index:Q')
    hline = alt.Chart(pd.DataFrame({'Clutch_Percentile':[0.5]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='Clutch_Percentile:Q')
    st.write("Chart data preview:", player_stats_df.head(), player_stats_df.shape)

    bubble_chart = alt.layer(bubble, vline, hline).properties(width=800, height=500, title='Player Consistency vs Clutchness').resolve_scale(x='shared', y='shared')
    st.altair_chart(bubble_chart, use_container_width=True, key=f"bubble00_{selected_tour}_{'-'.join(map(str, selected_years))}_{selected_tourney}")

    # ---- Top/Bottom Tables ----
    st.subheader("Most & Least Consistent Performance")
    col1, col2 = st.columns(2)
    with col1:
        render_flag_table(player_consistency_df.nlargest(10, 'Consistency Index'), player_col="Player", numeric_cols=["Consistency Index"])
    with col2:
        render_flag_table(player_consistency_df.nsmallest(10, 'Consistency Index'), player_col="Player", numeric_cols=["Consistency Index"])

    st.subheader("Most & Least Clutch Performance")
    if not match_clutch_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            render_flag_table(total_clutch_df.nlargest(10, 'Expected Points Added (EPA)'), player_col="Player", numeric_cols=["Expected Points Added (EPA)"])
        with col2:
            render_flag_table(total_clutch_df.nsmallest(10, 'Expected Points Added (EPA)'), player_col="Player", numeric_cols=["Expected Points Added (EPA)"])
    else:
        st.info("No clutch points/matches found.")
    print_memory("after rendering tab0")
    # After rendering bubble chart and tables
    del st.session_state.df_tab0, df_tab0, df_long, match_clutch_df, total_clutch_df, player_stats_df
    import gc; gc.collect()

with tab1:
    st.subheader("📊 Point Win Rates (All Points vs. High Pressure)")
    st.session_state.active_tab = "tab1"

    # ---- Lazy-load only columns needed for Tab 1 ----
    if "df_tab1" not in st.session_state or filters_changed:
        st.session_state.df_tab1 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
        st.session_state.df_tab1 = add_filtered_player_columns(st.session_state.df_tab1, selected_players)
    df_tab1 = st.session_state.df_tab1

    # df_tab1 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
    # df_tab1 = add_filtered_player_columns(df_tab1, selected_players)
    print_memory("after pulling in tab1 df")

    # ---- Add derived columns ----
    df_tab1["server_point_win"] = df_tab1["PointWinner"] == df_tab1["PointServer"]
    df_tab1["server_win"] = df_tab1["server_point_win"]
    df_tab1["returner_win"] = ~df_tab1["server_point_win"]

    # ---- Compute total points per player ----
    player_points = df_tab1.groupby('player1')['match_id'].count().reset_index().rename(
        columns={'match_id': 'Total_Points', 'player1': 'Player'}
    )
    df_tab1 = df_tab1.merge(player_points, left_on='player1', right_on='Player', how='left')
    df_tab1 = df_tab1[df_tab1['Total_Points'] >= min_points_filter]

    # ---- High Pressure Filters ----
    with st.expander("Configure High Pressure Filters"):
        pressure_threshold = st.slider(
            "Importance Threshold (Top N% of Point Probability +/-)",
            min_value=1, max_value=50, value=25, key="pressure_thr_tab1"
        )
        threshold_value = df_tab1["importance"].quantile(1 - pressure_threshold / 100)
        df_tab1["is_high_pressure"] = df_tab1["importance"] >= threshold_value

        max_hp_points = int(df_tab1["is_high_pressure"].sum())
        default_hp_points = min(200 if selected_tour=="ATP" else 100, max_hp_points)

        if max_hp_points > 0:
            min_hp_points_filter = st.slider(
                "Minimum High Pressure Points",
                min_value=0,
                max_value=2000,
                value=default_hp_points,
                key="min_hp_points_slider_tab1"
            )
        else:
            min_hp_points_filter = 0
            st.info("No high-pressure points in current selection")

    # ---- Compute High Pressure %
    player_hp = compute_high_pressure_pct(df_tab1, min_hp_points_filter)
    player_hp = player_hp.rename(columns={"player":"Player", "High_Pressure_Pct":"High Pressure %"})

    # ---- Compute Rankings ----
    rankings = compute_clutch_rankings(df_tab1).reset_index().rename(columns={
        "index": "Player",
        "Win % (All)": "Win % (All Points)",
        "Win % (HP)": "Win % (High Pressure)",
        "HP Points": "High Pressure Points"
    })

    rankings = rankings.merge(player_hp[["Player", "High Pressure %"]], on="Player", how="left")
    rankings = rankings.merge(player_points, on="Player", how="left")

    rankings_display_filtered = rankings[rankings["Total_Points"] >= min_points_filter].sort_values("Clutch Delta", ascending=False)
    rankings_display_filtered = rankings_display_filtered.rename(columns={"High Pressure %": "% High Pressure Points"})

    # ---- Bubble chart ----
    clutch_colors = alt.Scale(domain=[rankings_display_filtered['Clutch Delta'].min(), rankings_display_filtered['Clutch Delta'].max()],
                              range=['#ff4d4d', '#4dff4d'])
    x_min, x_max = rankings_display_filtered['Win % (All Points)'].min(), rankings_display_filtered['Win % (All Points)'].max()
    y_min, y_max = rankings_display_filtered['Win % (High Pressure)'].min(), rankings_display_filtered['Win % (High Pressure)'].max()

    bubble_chart = (
        alt.Chart(rankings_display_filtered)
        .mark_circle()
        .encode(
            x=alt.X('Win % (All Points):Q', scale=alt.Scale(domain=[x_min, x_max]), title='Total Win %'),
            y=alt.Y('Win % (High Pressure):Q', scale=alt.Scale(domain=[y_min, y_max]), title='Win % Under Pressure'),
            size=alt.Size('% High Pressure Points:Q', scale=alt.Scale(range=[10,500]), title='% Points Under Pressure'),
            color=alt.Color('Clutch Delta:Q', scale=clutch_colors, title='Clutch Delta'),
            tooltip=[
                alt.Tooltip('Player:N'),
                alt.Tooltip('Win % (All Points):Q', format=".1%"),
                alt.Tooltip('Win % (High Pressure):Q', format=".1%"),
                alt.Tooltip('% High Pressure Points:Q', format=".1%"),
                alt.Tooltip('Clutch Delta:Q', format=".1%")
            ]
        )
        .properties(width=800, height=500)
    )

    vline = alt.Chart(pd.DataFrame({'x':[0.5]})).mark_rule(color='gray', strokeDash=[4,4]).encode(x='x:Q')
    hline = alt.Chart(pd.DataFrame({'y':[0.5]})).mark_rule(color='gray', strokeDash=[4,4]).encode(y='y:Q')
    st.altair_chart(bubble_chart + vline + hline, use_container_width=True, key=f"bubble01_{selected_tour}_{'-'.join(map(str, selected_years))}_{selected_tourney}")


    # ---- Top/Bottom Tables ----
    st.subheader("Under Pressure Frequency")
    if not player_hp.empty:
        col1, col2 = st.columns(2)
        with col1:
            render_flag_table(player_hp.sort_values("High Pressure %").head(10), player_col="Player", numeric_cols=["High Pressure %"])
        with col2:
            render_flag_table(player_hp.sort_values("High Pressure %", ascending=False).head(10), player_col="Player", numeric_cols=["High Pressure %"])
    else:
        st.info("No pressure points found.")


    st.subheader("Point Win Rates Under Pressure")
    if not rankings_display_filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            render_flag_table(rankings_display_filtered.sort_values("Win % (High Pressure)", ascending=False).head(10), player_col="Player", numeric_cols=["Win % (High Pressure)"])
        with col2:
            render_flag_table(rankings_display_filtered.sort_values("Win % (High Pressure)").head(10), player_col="Player", numeric_cols=["Win % (High Pressure)"])
    else:
        st.info("No pressure points found.")
    print_memory("after rendering tab1")
    # Delete large intermediate DataFrames
    del st.session_state.df_tab1, df_tab1, player_points, player_hp, rankings, rankings_display_filtered, bubble_chart, vline, hline
    # Force garbage collection
    import gc; gc.collect()

# ---- TAB 2: Extreme Events ----
with tab2:
    st.subheader("🏆 Top 10 Most Unlikely Wins")
    st.session_state.active_tab = "tab2"

    columns = FEATURE_COLUMNS["base"] + FEATURE_COLUMNS["win_prob"] + FEATURE_COLUMNS["serve_return"] + FEATURE_COLUMNS["score"] + FEATURE_COLUMNS["match_winner"] 
    
    if "df_tab1" not in st.session_state or filters_changed:
        st.session_state.df_tab2 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter, columns)
        st.session_state.df_tab2 = add_filtered_player_columns(st.session_state.df_tab2, selected_players)
    df_tab2 = st.session_state.df_tab2

    # df_tab2 = load_filtered_df_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter, columns)
    # df_tab2 = add_filtered_player_columns(df_tab2, selected_players)
    print_memory("after pulling in tab2 df")

    if 'p1_win_prob_before' in df_tab2.columns and 'match_winner' in df_tab2.columns:
        df_valid = df_tab2[df_tab2['match_winner'].notna()]
        df_valid = filter_matches_by_sets(df_valid)

        # Compute pre-match win probability from the perspective of the actual winner
        df_valid['winner_prob_before'] = df_valid.apply(
            lambda r: r['p1_win_prob_before'] if r['match_winner'] == r['player1'] else 1 - r['p1_win_prob_before'], axis=1
        )

        # Split match_id for year/tourney/match number
        df_valid[['year', 'tourney_code', 'match_num']] = df_valid['match_id'].str.split('-', n=2, expand=True)
        df_valid['tournament_name'] = df_valid['tourney_code'].map(TOURNAMENTS_MAP).fillna(df_valid['tourney_code'])

        # Top 10 unlikely wins
        idxs = df_valid.groupby('match_id')['winner_prob_before'].idxmin()
        top_unlikely = df_valid.loc[idxs].sort_values('winner_prob_before').head(10)

        top_unlikely_display = top_unlikely[[
            'year', 'Tour', 'tournament_name', 'player1', 'player2', 'match_winner',
            'winner_prob_before', 'P1_Sets_Won', 'P2_Sets_Won', 'P1_Games_Won', 'P2_Games_Won', 'score', 'PointServer'
        ]].rename(columns={
            'year': 'Year', 'tournament_name': 'Tournament', 'player1': 'Player 1', 'player2': 'Player 2',
            'match_winner': 'Match Winner', 'winner_prob_before': 'Win Probability',
            'score': 'Game Score', 'P1_Sets_Won': 'P1 Sets', 'P2_Sets_Won': 'P2 Sets',
            'P1_Games_Won': 'P1 Games', 'P2_Games_Won': 'P2 Games', 'PointServer': 'Server'
        })

        top_unlikely_display['Win Probability'] *= 100  # format as percent

        # Render scoreboard in two columns
        if not top_unlikely_display.empty:
            rows_dicts = top_unlikely_display.to_dict(orient='records')
            half = len(rows_dicts) // 2 + len(rows_dicts) % 2
            left_rows, right_rows = rows_dicts[:half], rows_dicts[half:]
            col1, col2 = st.columns(2)

            for row in left_rows:
                with col1:
                    render_scoreboard(row)
            for row in right_rows:
                with col2:
                    render_scoreboard(row)
    else:
        st.warning("Required columns for 'Top 10 Most Unlikely Wins' not found.")

    # ---- Top 10 Highest Leverage Points ----
    st.subheader("🔥 Top 10 Highest Leverage Points (Largest Probability Swing)")

    top_points = compute_top_points(df_tab2, top_n=10)
    if not top_points.empty:
        # Prepare display
        top_points_display = top_points[[
            'year', 'Tour', 'tournament_name', 'player1', 'player2', 'score',
            'P1_Sets_Won', 'P2_Sets_Won', 'P1_Games_Won', 'P2_Games_Won',
            'wp_delta_display', 'PointWinnerName', 'match_winner',
            'p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins'
        ]].rename(columns={
            'year': 'Year', 'tournament_name': 'Tournament',
            'player1': 'Player 1', 'player2': 'Player 2', 'wp_delta_display': 'Prob. Swing',
            'score': 'Game Score', 'P1_Sets_Won': 'Sets 1', 'P2_Sets_Won': 'Sets 2',
            'P1_Games_Won': 'Games 1', 'P2_Games_Won': 'Games 2',
            'PointWinnerName': 'Point Winner', 'match_winner': 'Match Winner'
        })

        # Add auxiliary columns
        top_points_display = top_points_display.reset_index(drop=True)
        top_points_display['Point_Label'] = top_points_display.index.astype(str)
        top_points_display['prob_min'] = top_points_display[['p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins']].min(axis=1)
        top_points_display['prob_max'] = top_points_display[['p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins']].max(axis=1)
        top_points_display['prob_delta'] = top_points_display['prob_max'] - top_points_display['prob_min']
        top_points_display['Match Score'] = (
            top_points_display['Sets 1'].astype(str) + "-" + top_points_display['Sets 2'].astype(str)
            + ", " + top_points_display['Games 1'].astype(str) + "-" + top_points_display['Games 2'].astype(str)
        )
        top_points_display['Players'] = top_points_display['Player 1'] + " vs " + top_points_display['Player 2']
        top_points_display['Year & Tournament'] = top_points_display['Year'].astype(str) + " - " + top_points_display['Tournament']

        # Melt for Altair
        df_melt = top_points_display.melt(
            id_vars=['Point_Label', 'Prob. Swing', 'Player 1', 'Player 2', 'Players', 'Year & Tournament',
                     'Tournament', 'Match Score', 'Game Score', 'Point Winner', 'Match Winner'],
            value_vars=['p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins'],
            var_name='Scenario',
            value_name='Win Probability'
        )
        df_melt['Win Probability Percent'] = df_melt['Win Probability'] * 100

        # Map scenario to actual point outcome
        def get_point_color(row):
            if row['Scenario'] == 'p1_win_prob_if_p1_wins' and row['Point Winner'] == row['Player 1']:
                return 'Actual'
            elif row['Scenario'] == 'p1_win_prob_if_p2_wins' and row['Point Winner'] == row['Player 2']:
                return 'Actual'
            else:
                return 'Predicted'

        df_melt['Point Outcome'] = df_melt.apply(get_point_color, axis=1)

        # Altair charts
        labels_sorted = top_points_display.sort_values('prob_delta', ascending=False)['Point_Label'].tolist()
        chart_height = 500
        base = alt.Chart(df_melt).encode(y=alt.Y('Point_Label:N', sort=labels_sorted, axis=None))
        lines = alt.Chart(top_points_display).mark_rule(color='orange').encode(
            y=alt.Y('Point_Label:N', sort=labels_sorted, axis=None),
            x='prob_min:Q',
            x2='prob_max:Q',
            tooltip=[alt.Tooltip('Players:N', title='Match'),
                     alt.Tooltip('Year & Tournament:N', title='Tournament'),
                     alt.Tooltip('Match Score:N'),
                     alt.Tooltip('Game Score:N'),
                     alt.Tooltip('Point Winner:N'),
                     alt.Tooltip('Match Winner:N'),
                     alt.Tooltip('prob_delta:Q', title='Win Probability Delta', format=".1%")]
        )
        points = base.mark_circle(size=150).encode(
            x=alt.X('Win Probability:Q', title='Win Probability', axis=alt.Axis(format='.0%')),
            color=alt.Color('Point Outcome:N', scale=alt.Scale(domain=['Actual', 'Predicted'], range=['green','lightgray'])),
            tooltip=[alt.Tooltip('Players:N', title='Match'),
                     alt.Tooltip('Year & Tournament:N', title='Tournament'),
                     alt.Tooltip('Match Score:N'),
                     alt.Tooltip('Game Score:N'),
                     alt.Tooltip('Point Winner:N'),
                     alt.Tooltip('Match Winner:N'),
                     alt.Tooltip('Win Probability:Q', title='Win Probability', format=".1%")]
        )
        main_chart = (lines + points).properties(width=600, height=chart_height)
        left_labels = alt.Chart(top_points_display).mark_text(align='right', dx=-10).encode(
            y=alt.Y('Point_Label:N', sort=labels_sorted, axis=None),
            text='Player 1:N'
        ).properties(width=100, height=chart_height)
        right_labels = alt.Chart(top_points_display).mark_text(align='left', dx=10).encode(
            y=alt.Y('Point_Label:N', sort=labels_sorted, axis=None),
            text='Player 2:N'
        ).properties(width=100, height=chart_height)

        final_chart = alt.hconcat(left_labels, main_chart, right_labels)
        st.altair_chart(final_chart, use_container_width=True)

    else:
        st.info("No top impactful points found.")
    # After top unlikely wins are rendered
    del st.session_state.df_tab2, df_tab2, df_valid, top_unlikely, top_unlikely_display, top_points, top_points_display, df_melt, base, lines, points, final_chart
    import gc; gc.collect()
    print_memory("after rendering tab2")

st.session_state.last_filters = filters
