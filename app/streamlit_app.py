import streamlit as st  # Make sure this is at the very top
import altair as alt
import os
import duckdb
import pandas as pd

DUCKDB_FILE = "data/processed/sim_results.duckdb"
CSV_FILE = "data/processed/all_points_with_importance.csv"
TABLE_NAME = "importance_results"

def initialize_duckdb_from_csv():
    # Only do this if DuckDB file doesn't exist or table is missing
    need_init = False
    
    if not os.path.exists(DUCKDB_FILE):
        need_init = True
    else:
        con = duckdb.connect(DUCKDB_FILE)
        tables = con.execute("SHOW TABLES").fetchall()
        con.close()
        if (TABLE_NAME,) not in tables:
            need_init = True

    if need_init:
        st.write(f"Initializing DuckDB from CSV '{CSV_FILE}' — this might take a moment.")
        df_csv = pd.read_csv(CSV_FILE)
        con = duckdb.connect(DUCKDB_FILE)
        con.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM df_csv")
        con.close()
        st.write("DuckDB initialized successfully!")

# Call this once at the start
initialize_duckdb_from_csv()

@st.cache_data
def load_data():
    con = duckdb.connect(DUCKDB_FILE)
    df = con.execute(f"SELECT * FROM {TABLE_NAME}").df()
    con.close()
    return df

df = load_data()

# --- Extract gender based on match_id ---
df["gender"] = df["match_id"].astype(str).str.extract(r"(\d{4})$")[0].str[0]
df["gender"] = df["gender"].map({"1": "Men", "2": "Women"})

st.title("🎾 Tennis Pressure Simulation Explorer")
st.markdown("Analyze which players **thrive** or **struggle** under pressure.")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# 1. Tournament Filter
tournaments = ["All", "ausopen", "frenchopen", "wimbledon", "usopen"]
selected_tourney = st.sidebar.selectbox("Tournament", tournaments)

# 2. Gender Filter
genders = ["All", "Men", "Women"]
selected_gender = st.sidebar.selectbox("Gender", genders)

# 3. Serve/Return Filter
perspectives = ["Serve", "Return", "All"]
selected_perspective = st.sidebar.selectbox("Perspective", perspectives)

# 4. Best or Worst Performers
rank_type = st.sidebar.radio("Show", ["Top Performers", "Worst Performers"])

# 5. Pressure Threshold
pressure_threshold = st.sidebar.slider(
    "Top % of most important points considered high pressure",
    min_value=1, max_value=50, value=10
)

# --- Filter data ---
if selected_tourney != "All":
    df = df[df["tournament"] == selected_tourney]

if selected_gender != "All":
    df = df[df["gender"] == selected_gender]

# Define high-pressure points
threshold_value = df["importance"].quantile(1 - pressure_threshold / 100)
df["is_high_pressure"] = df["importance"] >= threshold_value

# Define point outcomes
df["server_point_win"] = (df["PointWinner"] == df["PointServer"])
df["server_win"] = df["server_point_win"]
df["returner_win"] = ~df["server_point_win"]

# Function to compute clutch stats
def compute_stats(df, player_col, win_col):
    def summarize(group):
        all_points = group[win_col]
        hp_points = group.loc[group["is_high_pressure"], win_col]

        return pd.Series({
            "Win % (All)": all_points.mean(),
            "Win % (HP)": hp_points.mean(),
            "Total Points": len(all_points),
            "HP Points": len(hp_points),
            "Clutch Delta": hp_points.mean() - all_points.mean()
        })

    summary = df.groupby(player_col).apply(summarize).dropna()
    summary = summary[summary["HP Points"] >= 20]
    return summary

# Apply correct perspective
if selected_perspective == "Serve":
    subset = df[df["server_name"].notna()]
    stats = compute_stats(subset, "server_name", "server_win")
elif selected_perspective == "Return":
    subset = df[df["returner_name"].notna()]
    stats = compute_stats(subset, "returner_name", "returner_win")
else:
    serve_stats = compute_stats(df[df["server_name"].notna()], "server_name", "server_win")
    return_stats = compute_stats(df[df["returner_name"].notna()], "returner_name", "returner_win")
    stats = pd.concat([serve_stats, return_stats])
    stats = stats.groupby(stats.index).agg({
        "Win % (All)": "mean",
        "Win % (HP)": "mean",
        "Total Points": "sum",
        "HP Points": "sum",
        "Clutch Delta": "mean"
    })

# Sort and display
ascending = rank_type == "Worst Performers"
top_stats = stats.sort_values("Clutch Delta", ascending=ascending).head(20)

st.subheader(f"📊 {rank_type} – {selected_perspective} ({selected_gender})")
st.dataframe(
    top_stats.style.format({
        "Win % (All)": "{:.2%}",
        "Win % (HP)": "{:.2%}",
        "Clutch Delta": "{:.2%}",
        "Total Points": "{:.0f}",
        "HP Points": "{:.0f}"
    })
)

# 🎯 Performance by Game Score

# Filter valid rows (remove NaNs if any)
score_df = df[df["score"].notna()].copy()

# Keep only normal tennis scores (no tiebreaks like "1-2")
valid_scores = ["0", "15", "30", "40", "AD"]
score_df = score_df[
    score_df["score"].str.extract(r"^([^-\s]+)-([^-\s]+)$")[0].isin(valid_scores) &
    score_df["score"].str.extract(r"^([^-\s]+)-([^-\s]+)$")[1].isin(valid_scores)
]

# Decide which win column to use based on perspective
if selected_perspective == "Serve":
    win_col = "server_win"
elif selected_perspective == "Return":
    win_col = "returner_win"
else:
    score_df["combined_win"] = (
        score_df["server_win"].astype(float) + score_df["returner_win"].astype(float)
    ) / 2
    win_col = "combined_win"

# Group by score
score_summary = (
    score_df.groupby("score")
    .agg(
        Average_Importance=("importance", "mean"),
        Win_Rate=(win_col, "mean"),
        Points=("importance", "count")
    )
    .sort_values("Points", ascending=False)
    .reset_index()
)

# Display with formatting
st.subheader("🎯 Performance by Game Score")
st.dataframe(
    score_summary.style.format({
        "Average_Importance": "{:.3f}",
        "Win_Rate": "{:.2%}",
        "Points": "{:,.0f}"
    })
)

st.subheader("🧱 Win Rate by Game Score Grid")

# Extract individual score components
df[["server_score_val", "returner_score_val"]] = df["score"].str.split("-", expand=True)

# Filter to valid rows
score_grid_df = df[df["server_score_val"].notna() & df["returner_score_val"].notna()].copy()

# Use selected perspective to choose win column
if selected_perspective == "Serve":
    win_col = "server_win"
elif selected_perspective == "Return":
    win_col = "returner_win"
else:
    score_grid_df["combined_win"] = (
        score_grid_df["server_win"].astype(float) + score_grid_df["returner_win"].astype(float)
    ) / 2
    win_col = "combined_win"

# Group and compute win rate
heatmap_data = (
    score_grid_df.groupby(["server_score_val", "returner_score_val"])[win_col]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "Win Rate", "count": "Points"})
)

# Convert scores to ordered categories for cleaner chart axes
ordered_scores = ["0", "15", "30", "40", "AD"]
heatmap_data = heatmap_data[
    heatmap_data["server_score_val"].isin(ordered_scores) &
    heatmap_data["returner_score_val"].isin(ordered_scores)
]
heatmap_data["server_score_val"] = pd.Categorical(heatmap_data["server_score_val"], categories=ordered_scores, ordered=True)
heatmap_data["returner_score_val"] = pd.Categorical(heatmap_data["returner_score_val"], categories=ordered_scores, ordered=True)

# Plot heatmap with Altair
heatmap = alt.Chart(heatmap_data).mark_rect().encode(
    x=alt.X("server_score_val:N", title="Server Score"),
    y=alt.Y("returner_score_val:N", title="Returner Score"),
    color=alt.Color("Win Rate:Q", scale=alt.Scale(scheme='viridis')),
    tooltip=["server_score_val", "returner_score_val", "Win Rate", "Points"]
).properties(
    width=400,
    height=400,
    title=f"{selected_perspective} Win Rate by Game Score"
)

st.altair_chart(heatmap, use_container_width=True)

# Function to get clutch stats for a player
def get_player_stats(player_name, df, perspective='Serve'):
    if perspective == 'Serve':
        player_df = df[df['server_name'] == player_name]
        win_col = 'server_win'
        player_col = 'server_name'
    else:
        player_df = df[df['returner_name'] == player_name]
        win_col = 'returner_win'
        player_col = 'returner_name'
    
    def summarize(group):
        all_points = group[win_col]
        hp_points = group.loc[group['is_high_pressure'], win_col]

        return pd.Series({
            "Win % (All)": all_points.mean(),
            "Win % (HP)": hp_points.mean(),
            "Total Points": len(all_points),
            "HP Points": len(hp_points),
            "Clutch Delta": hp_points.mean() - all_points.mean()
        })
    
    stats = summarize(player_df)
    return stats

