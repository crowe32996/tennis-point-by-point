# app.py
import streamlit as st
import pandas as pd
import numpy as np
from data import clean_player_name, load_tab0_sql, load_tab1_sql, load_tab2_sql
from utils import add_basic_columns, filter_matches_by_sets, apply_filters, add_filtered_player_columns, ioc_to_flag, add_flag_image, render_flag_table, load_player_flag_map, render_scoreboard
from data_processing import compute_player_deltas, compute_match_player_consistency, compute_clutch_rankings, compute_high_pressure_pct, compute_top_points, compute_unlikely_matches, compute_score_summary, compute_match_player_clutch, compute_player_clutch_aggregate
from streamlit.components.v1 import html as components_html
import altair as alt
import os
from transform import transform_tables
import duckdb
from pathlib import Path


# Base directory is the repo root
BASE_DIR = Path(__file__).resolve().parent.parent  # adjust if needed
if not (BASE_DIR / "data").exists():
    BASE_DIR = Path(__file__).resolve().parent  # fallback for cloud

DUCKDB_FILE = BASE_DIR / "outputs" / "sim_results.duckdb"


# --- Load data / mapping once ---
player_flag_map = st.cache_data(load_player_flag_map)()  # cached, read once per session


@st.cache_data
def get_sidebar_options():
    con = duckdb.connect(DUCKDB_FILE)
    tournaments = con.execute("SELECT DISTINCT tournament_name FROM match_detail").fetchall()
    tours = con.execute("SELECT DISTINCT tour FROM match_detail").fetchall()
    players = con.execute("SELECT DISTINCT player_status FROM player_detail").fetchall()
    years = con.execute("SELECT DISTINCT year FROM match_detail ORDER BY year").fetchall()

    con.close()

    # Flatten tuples and add "All"
    tournaments = ["All"] + [t[0] for t in tournaments]
    tours = [t[0] for t in tours]
    players = ["All"] + [p[0] for p in players]
    years = sorted([y[0] for y in years])

    return tournaments, tours, players, years

TOURNAMENTS_SIDEBAR, TOUR_SIDEBAR, PLAYERS_SIDEBAR, ALL_YEARS = get_sidebar_options()

# Year range
selected_year_range = st.sidebar.select_slider(
    "Select Year Range",
    options=ALL_YEARS,
    value=(min(ALL_YEARS), max(ALL_YEARS))
)
selected_years = list(range(selected_year_range[0], selected_year_range[1] + 1))

# Tour
selected_tour = st.sidebar.selectbox("Tour", TOUR_SIDEBAR, index=TOUR_SIDEBAR.index("ATP"))

# Tournament
selected_tourney = st.sidebar.selectbox("Tournament", TOURNAMENTS_SIDEBAR, index=TOURNAMENTS_SIDEBAR.index("All"))

# Player Status
selected_players = st.sidebar.selectbox("Player Status", PLAYERS_SIDEBAR, index=PLAYERS_SIDEBAR.index("Active"))

# Minimum points
default_min_points = (400 if selected_tour == "ATP" else 200) * len(selected_years)
min_points_filter = st.sidebar.slider(
    "Minimum Points per Player",
    min_value=0,
    max_value=5000,
    value=default_min_points,
    step=50
)


st.set_page_config(layout="wide", page_title="Tennis Historical Performance")
st.sidebar.header("Filters")


# print_memory("before any major ops")

# Dynamically update title
st.title(f"🎾 {selected_tour} Tennis Grand Slam Performance")
st.markdown(
    f"Analyze which players have **thrived** or **struggled** in high leverage points in {selected_tour} Grand Slam matches."
)


def load_tab_data(tab_key, columns=None):
    """Load tab dataframe if not in session_state or filters changed."""
    if tab_key not in st.session_state or st.session_state.get("reload_tabs", False):
        df = load_tab0_sql(
            selected_years,
            selected_tour,
            selected_tourney,
            selected_players,
            min_points_filter,
            columns
        )
        df = add_filtered_player_columns(df, selected_players)
        st.session_state[tab_key] = df
    return st.session_state[tab_key]

tab0_columns = [
    "match_id",
    "player1",
    "player2",
    "PointWinner",
    "p1_win_prob_before",
    "p1_win_prob_if_p1_wins",
    "p1_win_prob_if_p2_wins",
    "importance",
    "PointServer",
    "year",
    "tourney_code",
    "points_stake"   # only if filtering by tournament in the tab
]
# --------------------------
# Render Functions
# --------------------------
def render_tab0():
    st.subheader("📊 Player Consistency and Clutchness")
    df_tab0 = load_tab0_sql(selected_years,
            selected_tour,
            selected_tourney,
            selected_players,
            min_points_filter)
    df_tab0 = add_filtered_player_columns(df_tab0, selected_players)

    # print_memory("after pulling in tab0 df")

    # ---- Add derived columns ----
    df_tab0["server_point_win"] = df_tab0["point_winner"] == df_tab0["point_server"]
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
            player=df_tab0['player1'],
            player_wp_delta=df_tab0['p1_wp_delta'],
            wp_before_point=df_tab0['p1_win_prob_before']
        ),
        df_tab0.assign(
            player=df_tab0['player2'],
            player_wp_delta=df_tab0['p2_wp_delta'],
            wp_before_point=1 - df_tab0['p1_win_prob_before']
        )
    ], ignore_index=True)
    # ---- ADD THIS LINE ----
    df_long = df_long[df_long['player'].notna()]  # remove masked-out players

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

    bubble_chart = alt.layer(bubble, vline, hline).properties(width=800, height=500, title='Player Consistency vs Clutchness').resolve_scale(x='shared', y='shared')
    st.altair_chart(bubble_chart, use_container_width=True, key=f"bubble00_{selected_tour}_{'-'.join(map(str, selected_years))}_{selected_tourney}")

    # ---- Top/Bottom Tables ----
    st.subheader("Most & Least Consistent Performance")
    if not player_consistency_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            html = render_flag_table(
                player_consistency_df.nlargest(10, 'Consistency Index'),
                player_flag_map,
                player_col="Player",
                numeric_cols=["Consistency Index"]
            )
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            html = render_flag_table(
                player_consistency_df.nsmallest(10, 'Consistency Index'),
                player_flag_map,
                player_col="Player",
                numeric_cols=["Consistency Index"]
            )
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No consistency data found.")

    st.subheader("Most & Least Clutch Performance")
    if not match_clutch_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            html = render_flag_table(total_clutch_df.nlargest(10, 'Expected Points Added (EPA)'), player_flag_map, player_col="Player", numeric_cols=["Expected Points Added (EPA)"])
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            html = render_flag_table(total_clutch_df.nsmallest(10, 'Expected Points Added (EPA)'), player_flag_map, player_col="Player", numeric_cols=["Expected Points Added (EPA)"])
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No clutch points/matches found.")
    # print_memory("after rendering tab0")
    # After rendering bubble chart and tables
    del df_tab0, df_long, match_clutch_df, total_clutch_df, player_stats_df
    import gc; gc.collect()

def render_tab1():
    st.subheader("📊 Point Win Rates (All Points vs. High Pressure)")
    df_tab1 = load_tab1_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
    # print_memory("after pulling in tab1 df")
    df_tab1 = add_filtered_player_columns(df_tab1, selected_players)

    # ---- Add derived columns ----
    df_tab1["server_point_win"] = df_tab1["point_winner"] == df_tab1["point_server"]
    df_tab1["server_win"] = df_tab1["server_point_win"]
    df_tab1["returner_win"] = ~df_tab1["server_point_win"]

    # ---- Compute total points per player ----
    player_points = df_tab1.groupby('player1')['match_id'].count().reset_index().rename(
        columns={'match_id': 'Total_Points', 'player1': 'Player'}
    )
    df_tab1 = df_tab1.merge(player_points, left_on='player1', right_on='Player', how='left')
    df_tab1 = df_tab1[df_tab1['Total_Points'] >= min_points_filter]

    pressure_threshold = 25
    threshold_value = df_tab1["importance"].quantile(1 - pressure_threshold / 100)
    df_tab1["is_high_pressure"] = df_tab1["importance"] >= threshold_value

    max_hp_points = int(df_tab1["is_high_pressure"].sum())
    default_hp_points = min(200 if selected_tour=="ATP" else 100, max_hp_points)

    if selected_tour == "ATP":
        min_hp_points = 50 * len(selected_years)  # men: 50 points per year
    else:  # WTA
        min_hp_points = 30 * len(selected_years)  # women: 30 points per year

    # ---- Compute High Pressure %
    player_hp = compute_high_pressure_pct(df_tab1, min_hp_points)
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
            html = render_flag_table(player_hp.sort_values("High Pressure %").head(10), player_flag_map, player_col="Player", numeric_cols=["High Pressure %"])
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            html = render_flag_table(player_hp.sort_values("High Pressure %", ascending=False).head(10), player_flag_map, player_col="Player", numeric_cols=["High Pressure %"])
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No pressure points found.")


    st.subheader("Point Win Rates Under Pressure")
    if not rankings_display_filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            html = render_flag_table(rankings_display_filtered.sort_values("Win % (High Pressure)", ascending=False).head(10), player_flag_map, player_col="Player", numeric_cols=["Win % (High Pressure)"])
            st.markdown(html, unsafe_allow_html=True)
        with col2:
            html = render_flag_table(rankings_display_filtered.sort_values("Win % (High Pressure)").head(10), player_flag_map, player_col="Player", numeric_cols=["Win % (High Pressure)"])
            st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No pressure points found.")
    # print_memory("after rendering tab1")
    # Delete large intermediate DataFrames
    del df_tab1, player_points, player_hp, rankings, rankings_display_filtered, bubble_chart, vline, hline
    # Force garbage collection
    import gc; gc.collect()

def render_tab2():
    st.subheader("🏆 Top 10 Most Unlikely Wins")
    df_tab2 = load_tab2_sql(selected_years, selected_tour, selected_tourney, selected_players, min_points_filter)
    # print_memory("after pulling in tab2 df")

    if 'p1_win_prob_before' in df_tab2.columns and 'match_winner' in df_tab2.columns:
        df_valid = df_tab2[df_tab2['match_winner'].notna()]
        df_valid = filter_matches_by_sets(df_valid)

        # Compute pre-match win probability from the perspective of the actual winner
        df_valid['winner_prob_before'] = df_valid.apply(
            lambda r: r['p1_win_prob_before'] if r['match_winner'] == r['player1'] else 1 - r['p1_win_prob_before'], axis=1
        )

        # Split match_id for year/tourney/match number
        df_valid[['year', 'tourney_code', 'match_num']] = df_valid['match_id'].str.split('-', n=2, expand=True)

        # Top 10 unlikely wins
        idxs = df_valid.groupby('match_id')['winner_prob_before'].idxmin()
        top_unlikely = df_valid.loc[idxs].sort_values('winner_prob_before').head(10)

        top_unlikely_display = top_unlikely[[
            'year', 'tour', 'tournament_name', 'player1', 'player2', 'match_winner',
            'winner_prob_before', 'p1_sets_won', 'p2_sets_won', 'p1_games_won', 'p2_games_won', 'score', 'point_server'
        ]].rename(columns={
            'year': 'Year', 'tour':'Tour', 'tournament_name': 'Tournament', 'player1': 'Player 1', 'player2': 'Player 2',
            'match_winner': 'Match Winner', 'winner_prob_before': 'Win Probability',
            'score': 'Game Score', 'p1_sets_won': 'P1 Sets', 'p2_sets_won': 'P2 Sets',
            'p1_games_won': 'P1 Games', 'p2_games_won': 'P2 Games', 'point_server': 'Server'
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
                    html = render_scoreboard(row)
                    components_html(html)

            for row in right_rows:
                with col2:
                    html = render_scoreboard(row)
                    components_html(html)

    else:
        st.warning("Required columns for 'Top 10 Most Unlikely Wins' not found.")

    # ---- Top 10 Highest Leverage Points ----
    st.subheader("🔥 Top 10 Highest Leverage Points (Largest Probability Swing)")

    top_points = compute_top_points(df_tab2, top_n=10)
    if not top_points.empty:
        # Prepare display
        top_points_display = top_points[[
            'year', 'tour', 'tournament_name', 'player1', 'player2', 'score',
            'p1_sets_won', 'p2_sets_won', 'p1_games_won', 'p2_games_won',
            'wp_delta_display', 'PointWinnerName', 'match_winner',
            'p1_win_prob_if_p1_wins', 'p1_win_prob_if_p2_wins'
        ]].rename(columns={
            'year': 'Year', 'tour':'Tour', 'tournament_name': 'Tournament',
            'player1': 'Player 1', 'player2': 'Player 2', 'wp_delta_display': 'Prob. Swing',
            'score': 'Game Score', 'p1_sets_won': 'Sets 1', 'p2_sets_won': 'Sets 2',
            'p1_games_won': 'Games 1', 'p2_games_won': 'Games 2',
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
    del df_tab2, df_valid, top_unlikely, top_unlikely_display, top_points, top_points_display, df_melt, base, lines, points, final_chart
    import gc; gc.collect()
    # print_memory("after rendering tab2")

# --------------------------
# Tabs
# --------------------------


tab_labels = ["Player Performance", "High Pressure Performance", "Standout Events","Methodology"]
tabs = st.tabs(tab_labels)

# Track active tab manually
if "active_tab_index" not in st.session_state:
    st.session_state.active_tab_index = 0  # default to first tab

for i, tab in enumerate(tabs):
    with tab:
        if st.session_state.active_tab_index != i:
            st.session_state.active_tab_index = i
            st.session_state.reload_tabs = True  # optional: trigger reload
        # Render the tab content
        if i == 0:
            render_tab0()
        elif i == 1:
            render_tab1()
        elif i == 2:
            render_tab2()
        else:
            st.header("Methodology")
            st.markdown("**Overview**\n\nThis project aims to quantify subjective measures of consistency and clutchness of tennis players, using a win-probability model and Monte Carlo simulations at each point of all matches. Each point is simulated 3,000 times to estimate impact on match win probability.")
            st.subheader("Point Simulations")
            st.markdown("""
            For each recorded point, **3000 full match simulations** are calculated to determine each player's probability of winning the match. The 3000 simulations fall into 3 buckets, with 1000 simulations each to determine:
            - The probability *before the point*.  
            - The probability *if player 1 wins the point*.  
            - The probability *if player 2 wins the point*.  

            The difference associated with the match probabilities for either player winning the point is the point *importance*. 
            """)
            st.subheader("Key Metrics")
            st.markdown("""
            **Expected Points Added (Clutchness)**: This measure of clutchness is the estimated ATP/WTA points gained or lost compared to the player's normal level. It is calculated as a product of each players win probability added/lost per point played, the *importance* of those points, and the ATP/WTA points at stake in the given match.

            **Consistency**: This is a measure of how high a player's win probability is throughout all matches, and how stable those probabilities stay. The metric is calculated as the normalized product of the mean win probability added times the inverse of the standard deviation of the win probability change per point.

            **High Pressure Points**: Defined as the top quartile (top 25%) of points by *importance*.
            """)
            st.subheader("Assumptions")
            st.markdown("- Simulation count can be increased for increased accuracy, but 1,000 per scenario (3,000 per point) was used to weigh scale and accuracy.\n- Small-sample players may be noisy, so minimum total point threshold is set to 400 points per year selected for ATP, and 200 points per year selected for WTA.")
            st.subheader("Sources & Links")
            st.markdown(
                """
                - **Project Repository:** [Charlie Rowe - Tennis Consistency & Clutchness Analysis](https://github.com/crowe32996/tennis-point-by-point)
                - **Raw Data:** Point level data sourced from [Jeff Sackmann's Tennis Data Repository](https://github.com/JeffSackmann/tennis_slam_pointbypoint) .
                """
            )
