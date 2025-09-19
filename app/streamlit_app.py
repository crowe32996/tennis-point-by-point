# streamlit_app.py
import streamlit as st
import pandas as pd
import duckdb
import os
import altair as alt
from pathlib import Path

# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"
CSV_FILE = BASE_DIR / "outputs" / "all_points_with_importance.csv"
TABLE_NAME = "importance_results"
PLAYER_COUNTRY_FILE = BASE_DIR / "data" / "processed" / "player_countries.csv"
player_country_df = pd.read_csv(PLAYER_COUNTRY_FILE)
player_flag_map = dict(zip(player_country_df["player"], player_country_df["country"]))


TOURNAMENTS_MAP = {
    "ausopen": "Australian Open",
    "frenchopen": "French Open",
    "wimbledon": "Wimbledon",
    "usopen": "US Open"
}
TOURNAMENTS_SIDEBAR = ["All"] + list(TOURNAMENTS_MAP.values())

GENDERS = ["All", "Men", "Women"]
TOURS = ["All", "ATP", "WTA"]

@st.cache_data(show_spinner=False)
def load_df_from_duckdb():
    con = duckdb.connect(DUCKDB_FILE)
    df = con.execute(f"SELECT * FROM {TABLE_NAME}").df()
    con.close()
    return df

# -----------------------
# Helper functions
# -----------------------
def capitalize_name(name):
    parts = name.split()
    capitalized_parts = []
    for part in parts[1:]:
        # Handle both apostrophes and dashes
        for sep in ["'", "-"]:
            if sep in part:
                subparts = part.split(sep)
                subparts = [sp.capitalize() for sp in subparts]
                part = sep.join(subparts)
        capitalized_parts.append(part)
    return parts[0] + ' ' + ' '.join(capitalized_parts)

def add_basic_columns(df):
    # compute wp delta if missing
    if "p1_wp_delta" not in df.columns:
        df["p1_wp_delta"] = df.apply(
            lambda row: (row["p1_win_prob_if_p1_wins"] - row["p1_win_prob_before"])
            if row["PointWinner"] == 1
            else (row["p1_win_prob_if_p2_wins"] - row["p1_win_prob_before"]),
            axis=1
        )
        df["p2_wp_delta"] = -df["p1_wp_delta"]

    # gender extraction from match_id end (existing approach)
    # protect against missing values
    df["gender"] = df["match_id"].astype(str).str.extract(r"(\d{4})$")[0].str[0]
    df["gender"] = df["gender"].map({"1": "Men", "2": "Women"})
    # After adding the gender column
    df["Tour"] = df["gender"].map({"Men": "ATP", "Women": "WTA"})
    df["player1"] =df['player1'].apply(
        lambda x: x.split()[0] + ' ' + ' '.join([w.capitalize() for w in x.split()[1:]])
    )
    df["player2"]=df['player2'].apply(
        lambda x: x.split()[0] + ' ' + ' '.join([w.capitalize() for w in x.split()[1:]])
    )
    df["player1"] = df['player1'].apply(capitalize_name)
    df["player2"] = df['player2'].apply(capitalize_name)
    return df

def filter_matches_by_sets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid matches from the dataset:
      - Men: exclude if a player has 3 sets won or more
      - Women: exclude if a player has 2 sets won or more
    """
    def is_valid(group):
        gender = group["gender"].iloc[0]
        p1_sets = group["P1_Sets_Won"].max()
        p2_sets = group["P2_Sets_Won"].max()
        if gender == "Men" and (p1_sets >= 3 or p2_sets >= 3):
            return False
        if gender == "Women" and (p1_sets >= 2 or p2_sets >= 2):
            return False
        return True

    return df.groupby("match_id", group_keys=False).filter(is_valid).reset_index(drop=True)


def apply_filters(df, selected_tourney, selected_tour, selected_years):
    df2 = df.copy()
    df2["tournament"] = df2["tournament"].map(TOURNAMENTS_MAP).fillna(df2["tournament"])
    if selected_tourney != "All":
        df2 = df2[df2["tournament"] == selected_tourney]
    if selected_tour != "All":
        df2 = df2[df2["Tour"] == selected_tour]
    if selected_years:
        df2 = df2[df2["year"].isin(selected_years)]
    return df2

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
    "IRI": "IR"
    # add others as needed
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
                if col == "High Pressure %":
                    html += f"<td style='text-align:right'>{val*100:.1f}%</td>"
                else:
                    html += f"<td style='text-align:right'>{val:.3f}</td>"
            else:
                html += f"<td style='text-align:right'>{val}</td>"
        html += "</tr>"
    html += "</table></div>"

    st.markdown(html, unsafe_allow_html=True)

# -----------------------
# Cached computation functions (heavy things)
# -----------------------
@st.cache_data
def compute_player_deltas(df):
    # -------------------------
    # Build delta_df for both players
    # -------------------------
    p1_df = df[["player1", "p1_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player1": "player", "p1_wp_delta": "wp_delta"}
    )
    p2_df = df[["player2", "p2_wp_delta", "is_high_pressure", "Tour"]].rename(
        columns={"player2": "player", "p2_wp_delta": "wp_delta"}
    )

    delta_df = pd.concat([p1_df, p2_df], ignore_index=True)

    high_pressure_df = delta_df[delta_df["is_high_pressure"]]

    # -------------------------
    # Aggregate deltas
    # -------------------------
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

    # -------------------------
    # Merge all together
    # -------------------------
    player_delta_summary = player_delta_summary_all.merge(
        player_delta_summary_hp, on="player", how="left"
    ).fillna(0)

    return player_delta_summary


@st.cache_data
def compute_clutch_rankings(df, perspective="All", min_hp_points=20):
    # define server/return win columns
    df_local = df.copy()
    df_local["server_point_win"] = (df_local["PointWinner"] == df_local["PointServer"])
    df_local["server_win"] = df_local["server_point_win"]
    df_local["returner_win"] = (~df_local["server_point_win"]).astype(int)

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
    df_local = df.copy()
    
    # True swing: probability if point won minus probability if point lost
    df_local["wp_delta_display"] = df_local["p1_win_prob_if_p1_wins"] - df_local["p1_win_prob_if_p2_wins"]
    df_local["abs_wp_delta"] = df_local["wp_delta_display"].abs()
    df_local["PointWinnerName"] = df_local.apply(
        lambda row: row["player1"] if row["PointWinner"] == 1 else row["player2"],
        axis=1
    )
    top_points = df_local.nlargest(top_n, "abs_wp_delta").copy()
    
    # add tournament pretty name
    top_points[["year", "tourney_code", "match_num"]] = top_points["match_id"].str.split("-", n=2, expand=True)
    top_points["tournament_name"] = top_points["tourney_code"].map(TOURNAMENTS_MAP).fillna(top_points["tourney_code"])
    return top_points

@st.cache_data
def compute_unlikely_matches(df, low_threshold=0.10, high_threshold=0.90):
    # For unlikely winners & unlikely losers counts (one per match, per player)
    df_valid = df[df["match_winner"].notna()].copy()
    df_valid = filter_matches_by_sets(df_valid)

    # Build p1/p2 version
    p1_df = df_valid.copy()
    p1_df["win_prob"] = p1_df["p1_win_prob_before"]
    p1_df["is_winner"] = p1_df["match_winner"] == p1_df["player1"]
    p1_df["player"] = p1_df["player1"]

    p2_df = df_valid.copy()
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
def compute_match_swings(df):
    """
    Returns one row per match with:
    - Year
    - Tournament (pretty)
    - Winner
    - Loser
    - Winner swing (max ATP/WTA Points Gained/Lost swing)
    - Winner Low Probability (lowest win probability at any point)
    - ATP Points at Stake
    - Gender
    """
    results = []
    
    for match_id, group in df.groupby("match_id"):
        grp = group.copy()
        
        # parse year and tournament
        parts = str(match_id).split("-", 2)
        year = parts[0] if len(parts) > 0 else None
        tourney_code = parts[1] if len(parts) > 1 else None
        tournament_name = TOURNAMENTS_MAP.get(tourney_code, tourney_code)
        
        # compute cumulative swing
        grp["p1_expected_points"] = grp["p1_win_prob_before"] * grp["points_stake"]
        grp["p2_expected_points"] = (1 - grp["p1_win_prob_before"]) * grp["points_stake"]
        
        # swing = max - min
        p1_swing = grp["p1_expected_points"].max() - grp["p1_expected_points"].min()
        p2_swing = grp["p2_expected_points"].max() - grp["p2_expected_points"].min()
        
        # determine winner/loser
        winner_name = grp["match_winner"].iloc[0] if "match_winner" in grp.columns else None
        if winner_name == grp["player1"].iloc[0]:
            winner_swing = p1_swing
            winner_low_prob = grp["p1_win_prob_before"].min()
            loser_name = grp["player2"].iloc[0]
        else:
            winner_swing = p2_swing
            winner_low_prob = (1 - grp["p1_win_prob_before"]).min()
            loser_name = grp["player1"].iloc[0]
        
        results.append({
            "Year": int(year) if year and year.isdigit() else None,
            "Tournament": tournament_name,
            "Winner": winner_name,
            "Loser": loser_name,
            "ATP/WTA Points Gained/Lost": winner_swing,
            "Winner Low Probability": winner_low_prob,
            "ATP Points at Stake": grp["points_stake"].iloc[0],
            "Tour": grp["Tour"].iloc[0] if "Tour" in grp.columns else "Unknown",
            "match_id": match_id
        })
    
    # sort by winner swing descending so biggest comebacks/blown leads appear first
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("ATP/WTA Points Gained/Lost", ascending=False).reset_index(drop=True)
    return df_results

@st.cache_data
def compute_score_summary(df):
    score_df = df[df["score"].notna()].copy()
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
    results = []

    # Ensure 'tourney_code' exists
    if "tourney_code" not in df.columns:
        df["tourney_code"] = df["match_id"].astype(str).str.split("-", n=2).str[1]

    # Iterate over matches
    for match_id, group in df.groupby("match_id"):
        tourney_code = group["tourney_code"].iloc[0]  # <-- get tourney_code for this match
        players = [group["player1"].iloc[0], group["player2"].iloc[0]]
        
        for player in players:
            # Determine which WP delta column corresponds to this player
            group = group.copy()
            group["player_wp_delta"] = group.apply(
                lambda row: row["p1_wp_delta"] if player == row["player1"] else row["p2_wp_delta"], axis=1
            )

            # Compute clutch score per point
            group["clutch_score"] = group["player_wp_delta"] * group["importance"] * group["points_stake"]

            # Aggregate per match
            total_clutch = group["clutch_score"].sum()
            hp_points = group["is_high_pressure"].sum()

            results.append({
                "match_id": match_id,
                "player": player,
                "Total_Clutch_Score": total_clutch,
                "High_Pressure_Points": hp_points,
                "tourney_code": tourney_code
            })

    return pd.DataFrame(results)

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

def make_score_heatmap(df):
    # expects df with server_score_val & returner_score_val & win_col in caller; we'll take from caller instead
    # Provide a simple placeholder to avoid errors if not used
    return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rect()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(layout="wide", page_title="Tennis Clutch Dashboard")

st.title("🎾 ATP/WTA Tennis Clutch Performers")
st.markdown("Analyze which players have **thrived** or **struggled** under pressure in Grand Slam matches since 2020.")

# load raw DF (cached)
raw_df = load_df_from_duckdb()
raw_df = add_basic_columns(raw_df)

# Extract year for filtering (ensure integer type)
raw_df["year"] = raw_df["match_id"].astype(str).str.split("-").str[0].astype(int)
available_years = sorted(raw_df["year"].unique())

# Sidebar filters (one place)
st.sidebar.header("Filters")
selected_year_range = st.sidebar.select_slider(
    "Select Year Range",
    options=available_years,
    value=(min(available_years), max(available_years))  # dynamically fits your data
)
# Convert the slider range tuple into a list of years
selected_years = list(range(selected_year_range[0], selected_year_range[1] + 1))
#selected_gender = st.sidebar.selectbox("Gender", GENDERS, index=0, key="gender_select")
selected_tour = st.sidebar.selectbox("Tour", TOURS, index=0, key="tour_select")
selected_tourney = st.sidebar.selectbox("Tournament", TOURNAMENTS_SIDEBAR, index=0, key="tourney_select")
pressure_threshold = st.sidebar.slider("Importance Threshold (Top N% of Point Probability +/-)", min_value=1, max_value=50, value=25, key="pressure_thr")

# apply filters to df used by many pages
df = apply_filters(raw_df, selected_tourney, selected_tour, selected_years)

if "is_high_pressure" not in df.columns:
    threshold_value = df["importance"].quantile(1 - pressure_threshold / 100)
    df["is_high_pressure"] = df["importance"] >= threshold_value

# Sidebar: minimum high pressure points
max_hp_points = int(df["is_high_pressure"].sum())


# default to 200 or max available if less than 200
default_hp_points = min(200, max_hp_points)

# show slider only if there are any high-pressure points
if max_hp_points > 0:
    min_hp_points_filter = st.sidebar.slider(
        "Minimum High Pressure Points", 
        min_value=0, 
        max_value=2000,
        value=default_hp_points,
        key="min_hp_points_slider"
    )
else:
    min_hp_points_filter = 0
    st.sidebar.info("No high-pressure points in current selection")

# compute HP % per player
hp_pct_df = compute_high_pressure_pct(df, min_hp_points_filter)

df["server_point_win"] = (df["PointWinner"] == df["PointServer"])
df["server_win"] = df["server_point_win"]
df["returner_win"] = ~df["server_point_win"]


# -----------------------
# Tabs: create them
# -----------------------
tab0, tab1, tab2, tab3 = st.tabs(["Clutch Summary", "Clutch Player Stats", "Extreme Events", "Game Score Stats"])

# ---- TAB 1: Clutch Player Rankings ----
# ---- TAB 1: Clutch Player Rankings ----
# ---- TAB 1: Clutch Player Rankings ----
with tab0:
    st.subheader("Players with Most & Least Clutch Performance")

    # Use the total clutch ranking from tab1
    # Compute total clutch per player (if not already available)
    match_clutch_df = compute_match_player_clutch(df)
    if not match_clutch_df.empty:
        total_clutch = match_clutch_df.groupby("player")["Total_Clutch_Score"].sum().reset_index()
        total_clutch = total_clutch.rename(columns={
            "player": "Player",
            "Total_Clutch_Score": "Total Clutch Score"
        })

        # Sort descending and ascending
        clutch_desc = total_clutch.sort_values("Total Clutch Score", ascending=False).head(10)
        clutch_asc = total_clutch.sort_values("Total Clutch Score", ascending=True).head(10)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 10 Most Clutch Players**")
            render_flag_table(
                clutch_desc.reset_index(drop=True),
                player_col="Player",
                numeric_cols=["Total Clutch Score"]
            )
        with col2:
            st.markdown("**Top 10 Least Clutch Players**")
            render_flag_table(
                clutch_asc.reset_index(drop=True),
                player_col="Player",
                numeric_cols=["Total Clutch Score"]
            )
    else:
        st.info("No clutch points/matches found.")

    player_hp = compute_high_pressure_pct(df, min_hp_points_filter)
    player_hp = player_hp.rename(columns={
        "player": "Player",
        "High_Pressure_Pct": "High Pressure %"
    })

    st.subheader("Percentage of Points Under Pressure")
    col1, col2 = st.columns(2)

    # Sort by High Pressure %
    hp_sorted_asc = player_hp.sort_values("High Pressure %", ascending=True)
    hp_sorted_desc = player_hp.sort_values("High Pressure %", ascending=False)

    with col1:
        st.markdown("**Top 10 Players Least Under Pressure**")
        render_flag_table(
            hp_sorted_asc[["Player", "High Pressure %"]].head(10).reset_index(drop=True),
            player_col="Player",
            numeric_cols=["High Pressure %"]
        )

    with col2:
        st.markdown("**Top 10 Players Most Under Pressure**")
        render_flag_table(
            hp_sorted_desc[["Player", "High Pressure %"]].head(10).reset_index(drop=True),
            player_col="Player",
            numeric_cols=["High Pressure %"]
        )

    st.subheader("Players with Most Unlikely Outcomes")
    wins_summary, losses_summary = compute_unlikely_matches(df, low_threshold=0.25, high_threshold=0.75)

    wins_summary = wins_summary.rename(columns={
        "player": "Player",
        "Num_UnlikelyWins": "# of Unlikely Wins"
    })
    losses_summary = losses_summary.rename(columns={
        "player": "Player",
        "Num_UnlikelyLosses": "# of Unlikely Losses"
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 10 Most Unlikely Wins (<25%)**")
        render_flag_table(
            wins_summary.reset_index(drop=True).head(10),
            player_col="Player",
            numeric_cols=["# of Unlikely Wins"]
        )

    with col2:
        st.markdown("**Top 10 Most Unlikely Losses (>75%)**")
        render_flag_table(
            losses_summary.reset_index(drop=True).head(10),
            player_col="Player",
            numeric_cols=["# of Unlikely Losses"]
        )


with tab1:
    st.subheader("📈 Player Clutch Scores (Best & Worst Matches)")

    # Compute match-level clutch
    match_clutch_df = compute_match_player_clutch(df)

    if not match_clutch_df.empty:
        # Keep only match_id, player, and Total_Clutch_Score
        match_clutch_df = match_clutch_df[["match_id", "player", "Total_Clutch_Score", "tourney_code"]]

        # Merge basic match info
        match_clutch_df = match_clutch_df.merge(
            df[["match_id", "Tour","year", "player1", "player2"]].drop_duplicates(),
            on="match_id",
            how="left"
        )

        # Map tournament codes to friendly names
        match_clutch_df["Tournament"] = match_clutch_df["tourney_code"].map(TOURNAMENTS_MAP).fillna(match_clutch_df["tourney_code"])

        # Combine year and tournament (year first)
        match_clutch_df["Tournament_Year"] = match_clutch_df["year"].astype(str) + " " + match_clutch_df["Tournament"]

        # Determine the opponent
        match_clutch_df["Opponent"] = match_clutch_df.apply(
            lambda r: r["player2"] if r["player"] == r["player1"] else r["player1"], axis=1
        )

        # Find each player's most and least clutch matches
        idx_max = match_clutch_df.groupby("player")["Total_Clutch_Score"].idxmax()
        idx_min = match_clutch_df.groupby("player")["Total_Clutch_Score"].idxmin()

        best_matches = match_clutch_df.loc[idx_max].rename(columns={
            "Total_Clutch_Score": "Best Clutch Score",
            "Tournament_Year": "Tournament (Most Clutch)",
            "Opponent": "Opponent (Most Clutch)"
        })[["player","Tour", "Best Clutch Score", "Tournament (Most Clutch)", "Opponent (Most Clutch)"]]

        worst_matches = match_clutch_df.loc[idx_min].rename(columns={
            "Total_Clutch_Score": "Worst Clutch Score",
            "Tournament_Year": "Tournament (Least Clutch)",
            "Opponent": "Opponent (Least Clutch)"
        })[["player", "Worst Clutch Score", "Tournament (Least Clutch)", "Opponent (Least Clutch)"]]

        # Merge best and worst
        player_matches = best_matches.merge(
            worst_matches,
            on="player"
        )

        # Add total clutch score across all matches per player
        total_clutch = match_clutch_df.groupby("player")["Total_Clutch_Score"].sum().reset_index()
        
        total_clutch = total_clutch.rename(columns={"Total_Clutch_Score": "Total Clutch Score"})

        player_matches = player_matches.merge(total_clutch, on="player", how="left")

        # Reorder columns
        player_matches = player_matches[[
            "Tour",
            "player",
            "Total Clutch Score",
            "Best Clutch Score",
            "Tournament (Most Clutch)",
            "Opponent (Most Clutch)",
            "Worst Clutch Score",
            "Tournament (Least Clutch)",
            "Opponent (Least Clutch)"
        ]]

        # Rename player column
        player_matches = player_matches.rename(columns={"player": "Player"})
        player_matches = player_matches.sort_values(by="Total Clutch Score", ascending=False)
        st.dataframe(
            player_matches.reset_index(drop=True).style.format({
                "Total Clutch Score": "{:.3f}",
                "Best Clutch Score": "{:.3f}",
                "Worst Clutch Score": "{:.3f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No clutch points/matches found.")

    st.subheader("📊 Points Win Rate (All Points vs. High Pressure)")
    
    # Compute rankings
    rankings = compute_clutch_rankings(df, perspective="All").reset_index().rename(columns={
        "index": "Player",
        "Win % (All)": "Win % (All Points)",
        "Win % (HP)": "Win % (High Pressure)",
        "HP Points": "High Pressure Points"
    })

    # Merge high-pressure percentage from hp_pct_df
    rankings = rankings.merge(
        hp_pct_df[["player", "High_Pressure_Pct"]],
        left_on="Player",
        right_on="player",
        how="left"
    )
    rankings.drop(columns="player", inplace=True)

    # Sort by Clutch Delta descending
    rankings_display_filtered = rankings[rankings["High Pressure Points"] >= min_hp_points_filter].copy()
    rankings_display_filtered = rankings_display_filtered.sort_values("Clutch Delta", ascending=False).rename(columns={
        "High_Pressure_Pct": "% High Pressure Points"
    })

    # Columns to display
    numeric_cols = [
        "Win % (All Points)", "Win % (High Pressure)", "Clutch Delta",
        "Total Points", "High Pressure Points", "% High Pressure Points"
    ]

    # Display sortable dataframe
    st.dataframe(
        rankings_display_filtered[["Tour","Player"] + numeric_cols].reset_index(drop=True).style.format({
            "Win % (All Points)": "{:.1%}",
            "Win % (High Pressure)": "{:.1%}",
            "Clutch Delta": "{:.1%}",
            "% High Pressure Points": "{:.1%}",
            "Total Points": "{:.0f}",
            "High Pressure Points": "{:.0f}"
        }),
        use_container_width=True
    )


# ---- TAB 2: Extreme Events ----
with tab2:
    st.subheader("🏆 Top 10 Most Unlikely Wins")
    if "p1_win_prob_before" in df.columns and "match_winner" in df.columns:
        df_valid = df[df["match_winner"].notna()].copy()
        df_valid = filter_matches_by_sets(df_valid)
        df_valid["winner_prob_before"] = df_valid.apply(
            lambda r: r["p1_win_prob_before"] if r["match_winner"] == r["player1"] else 1 - r["p1_win_prob_before"],
            axis=1
        )
        df_valid[["year", "tourney_code", "match_num"]] = df_valid["match_id"].str.split("-", n=2, expand=True)
        df_valid["tournament_name"] = df_valid["tourney_code"].map(TOURNAMENTS_MAP).fillna(df_valid["tourney_code"])
        idxs = df_valid.groupby("match_id")["winner_prob_before"].idxmin()
        top_unlikely = df_valid.loc[idxs].sort_values("winner_prob_before").head(10)
        top_unlikely_display = top_unlikely[[
            "year", "Tour", "tournament_name", "player1", "player2", "match_winner",
            "winner_prob_before", "P1_Sets_Won", "P2_Sets_Won", "P1_Games_Won", "P2_Games_Won", "score"
        ]].rename(columns={
            "year": "Year", "tournament_name": "Tournament", "player1": "Player 1", "player2": "Player 2",
            "match_winner": "Match Winner", "winner_prob_before": "Win Probability",
            "score": "Game Score", "P1_Sets_Won": "P1 Sets", "P2_Sets_Won": "P2 Sets",
            "P1_Games_Won": "P1 Games", "P2_Games_Won": "P2 Games"
        })
        # format probability as percent
        top_unlikely_display["Win Probability"] = top_unlikely_display["Win Probability"] * 100
        st.dataframe(top_unlikely_display.reset_index(drop=True).style.format({"Win Probability": "{:.1f}%"}), use_container_width=True)
    else:
        st.warning("Required columns for 'Top 10 Most Unlikely Wins' not found (p1_win_prob_before / match_winner).")

    st.subheader("🔥 Top 10 Most Impactful Points")
    top_points = compute_top_points(df, top_n=10)
    if not top_points.empty:
        # create a display frame
        top_points_display = top_points[[
            "year", "Tour", "tournament_name", "player1", "player2", "score", "P1_Sets_Won", "P2_Sets_Won",
            "P1_Games_Won", "P2_Games_Won", "wp_delta_display", 
            #"p1_win_prob_before", "p1_win_prob_if_p1_wins", "p1_win_prob_if_p2_wins", 
            "PointWinnerName", "match_winner"
        ]].rename(columns={
            "year": "Year", "tournament_name": "Tournament", "player1": "Player 1", "player2": "Player 2",
            "wp_delta_display": "Prob. Swing", 
            #"p1_win_prob_before": "Player1 Win Probability pre-point", "p1_win_prob_if_p1_wins": "Player1 Win Prob if Won Point", "p1_win_prob_if_p2_wins": "Player1 Win Prob if Lost Point",
            "score": "Score", "P1_Sets_Won": "Sets_1", "P2_Sets_Won": "Sets_2",
            "P1_Games_Won": "Games_1", "P2_Games_Won": "Games_2", "PointWinnerName":"Point Winner", "match_winner":"Match Winner"
        })
        st.dataframe(top_points_display.reset_index(drop=True).style.format({
            "Prob. Swing": "{:.1%}"
            #"Player1 Win Probability pre-point": "{:.1%}",
            #"Player1 Win Prob if Won Point": "{:.1%}",
            #"Player1 Win Prob if Lost Point": "{:.1%}"
        }), use_container_width=True)
    else:
        st.info("No top impactful points found.")
    st.subheader("💪 Top 10 Most Clutch Matches")

    # Compute clutch if your function exists
    clutch_df = compute_match_player_clutch(df)

    if not clutch_df.empty:
        # Aggregate by match and player
        clutch_agg = (
            clutch_df.groupby(["match_id", "player"], as_index=False)
            .agg({"Total_Clutch_Score": "sum"})
        )
        print(df.columns.tolist())
        st.write("Columns in df:", df.columns.tolist())

        # Merge basic match info (year, player1, player2)
        clutch_agg = clutch_agg.merge(
            df[["match_id", "year", "Tour","tourney_code","player1", "player2"]].drop_duplicates(),
            on="match_id",
            how="left"
        )

        # Map tournament codes to friendly names
        clutch_agg["Tournament"] = clutch_agg["tourney_code"].map(TOURNAMENTS_MAP).fillna(clutch_agg["tourney_code"])

        # Sort by total clutch score and take top 10
        clutch_agg = clutch_agg.sort_values("Total_Clutch_Score", ascending=False).head(10)

        clutch_display = clutch_agg.rename(columns={
            "player": "Match Winner",
            "Total_Clutch_Score": "Clutch Score",
            "year": "Year",
            "player1": "Player 1",
            "player2": "Player 2"
        })[["Year", "Tour","Tournament", "Player 1", "Player 2", "Match Winner", "Clutch Score"]]

        st.dataframe(
            clutch_display.reset_index(drop=True).style.format({"Clutch Score": "{:.3f}"}),
            use_container_width=True
        )
    else:
        st.info("No clutch points/matches found.")

                
    st.subheader("📈 Most Significant Comebacks / Blown Leads")
    swings_df = compute_match_swings(df)
    if not swings_df.empty:
        # convert win probability to percent for display
        swings_df["Winner Low Probability %"] = swings_df["Winner Low Probability"] * 100
        display_cols = ["Year", "Tour", "Tournament", "Winner", "Loser", "ATP/WTA Points Gained/Lost", "Winner Low Probability %", "ATP Points at Stake"]
        st.dataframe(swings_df[display_cols].rename(columns={"Winner Low Probability %": "Winner Low Probability"}).reset_index(drop=True).style.format({
            "ATP/WTA Points Gained/Lost": "{:.1f}",
            "Winner Low Probability": "{:.1f}%",
            "ATP Points at Stake": "{:.0f}"
        }), use_container_width=True)
    else:
        st.info("No swings data returned.")
# ---- TAB 3: Summary Statistics ----
with tab3:
    st.subheader("⚡Server Performance by Game Score")
    score_summary = compute_score_summary(df)
    if not score_summary.empty:
        st.dataframe(
            score_summary.rename(columns={
                "score": "Game Score",
                "Average_Importance": "Average Probability Swing",
                "Win_Rate": "Win Rate",
                "Points": "Total Points"
            }).reset_index(drop=True).style.format({
                "Average Probability Swing": "{:.1%}",
                "Win Rate": "{:.2%}",
                "Total Points": "{:,.0f}"
            }),
            use_container_width=True
        )
    else:
        st.info("No score summary available.")

    st.subheader("🧱 Win Rate by Game Score Grid")

    # --- Compute numeric win columns ---
    df["server_win"] = (df["PointWinner"] == df["PointServer"]).astype(int)
    df["returner_win"] = (df["PointWinner"] != df["PointServer"]).astype(int)

    # --- Extract individual score components ---
    df[["server_score_val", "returner_score_val"]] = df["score"].str.split("-", expand=True)

    # --- Filter to valid rows ---
    valid_scores = ["0", "15", "30", "40", "AD"]
    score_grid_df = df[
        df["server_score_val"].isin(valid_scores) &
        df["returner_score_val"].isin(valid_scores)
    ].copy()

    score_grid_df["win_numeric"] = score_grid_df["server_win"]

    # --- Group by score and compute win rate & point counts ---
    heatmap_data = (
        score_grid_df.groupby(["server_score_val", "returner_score_val"], dropna=False)["win_numeric"]
        .agg(Win_Rate="mean", Points="count")
        .reset_index()
    )

    # --- Convert to ordered categories for axes ---
    score_order = ["0", "15", "30", "40", "AD"]
    heatmap_data["server_score_val"] = pd.Categorical(
        heatmap_data["server_score_val"], categories=score_order, ordered=True
    )
    heatmap_data["returner_score_val"] = pd.Categorical(
        heatmap_data["returner_score_val"], categories=score_order, ordered=True
    )

    # --- Altair heatmap ---
    heatmap = alt.Chart(heatmap_data).mark_rect().encode(
        x=alt.X("server_score_val:N", title="Server Score"),
        y=alt.Y("returner_score_val:N", title="Returner Score"),
        color=alt.Color("Win_Rate:Q", scale=alt.Scale(scheme="viridis"), title="Win Rate"),
        tooltip=[
            "server_score_val",
            "returner_score_val",
            alt.Tooltip("Win_Rate", format=".2%"),
            "Points"
        ]
    ).properties(
        width=400,
        height=400,
        title="Win Rate by Game Score"
    )

    st.altair_chart(heatmap, use_container_width=True)

